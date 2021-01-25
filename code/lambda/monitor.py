import json
import boto3
import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
client = boto3.client('sagemaker')
sns = boto3.client('sns')

def lambda_handler(event, context):
    """ Monitor active run, stop if issuesFound, and define retraining job. """
    try:
        job_name = event["state"]["history"]["latest_job_name"]
        logger.info(f'Job name: {job_name}')
        topic_arn = event['topic_arn']
        max_num_retraining = event['max_num_retraining']
        max_monitor_transitions = event['max_monitor_transitions']
    except KeyError as e:
        raise KeyError('Bad input.' +
                       'Key error is : {} \n'.format(e) +
                       'The input received was: {}'.format(event)
                       )

    state = event.get("state")
    state['history']['num_monitor_transitions'] += 1

    try:
        job_description = client.describe_training_job(TrainingJobName=job_name)
    except Exception as e:
        logger.error(
            "Encountered error while trying to "
            "stop training job {}: {}".format(
                job_name, str(e)
            )
        )
        raise e

    job_status = job_description.get('TrainingJobStatus')
    state["job_status"] = job_status
    logger.info(f'Job status: {job_status}')

    if state['history']['num_monitor_transitions'] > max_monitor_transitions:
        # stop the machine, too many transitions
        sns.publish(TopicArn=topic_arn,
                    Message=f'Ending. Number of monitor transitions {state["history"]["num_monitor_transitions"]} > {max_monitor_transitions}')
        stop_job(job_name)
        return {
            'statusCode': 200,
            'body': {
                'state': state
            }
        }

    if job_status == 'Completed':
        state["job_status"] = job_status
        # making sure we don't have any lingering debugger jobs before we terminating the machine
        stop_processing_job(job_description['DebugRuleEvaluationStatuses'][0]['RuleEvaluationJobArn'].split('/')[-1])   
        state["next_action"] = "end"
        sns.publish(TopicArn=topic_arn, Message=f'Job completed. Final state is: {state}. Final training job name is {job_name}.')
        return {
            'statusCode': 200,
            'body': {
                'state': state
            }
        }
    elif job_status == 'Failed':
        state["job_status"] = "Failed"
        reason = job_description.get('FailureReason', '')
        if 'AlgorithmError' not in reason:
            state["next_action"] = "launch_new"
            sns.publish(TopicArn=topic_arn, Message=f'Training failed with reason: {reason}\n'
                                                    f'Job description: {job_description}\n'
                                                    f'Machine state: {state}')
        else:
            state["next_action"] = "end"
            
    else:
        rules_eval_statuses = job_description.get('DebugRuleEvaluationStatuses', None)
        if rules_eval_statuses is None or len(rules_eval_statuses) == 0:
            logger.info("Couldn't find any debug rule statuses, skipping...")
            state["rule_status"] = "NotFound"
            state["next_action"] = "monitor"
            return {
                'statusCode': 200,
                'body': {
                    'state': state
                }
            }
        rule = rules_eval_statuses[0]
        if rule['RuleEvaluationStatus'] == "IssuesFound":
            logging.info(
                'Evaluation of rule configuration {} resulted in "IssuesFound". '
                'Attempting to stop training job {}'.format(
                    rule.get("RuleConfigurationName"), job_name
                )
            )
            stop_job(job_name)
            logger.info('Planning a new launch')
            state = plan_launch_spec(state)
            logger.info(f'New training spec {json.dumps(state["run_spec"])}')
            state["rule_status"] = "ExplodingTensors"
        elif rule['RuleEvaluationStatus'] == "InProgress":
            logger.info(
                'Evaluation of rule configuration {} of job {} is in progress. '.format(
                    rule.get("RuleConfigurationName"), job_name
                )
            )
            state["rule_status"] = "InProgress"
            state["next_action"] = "monitor"
        else:
            logging.info(
                'Status of rule configuration {} of job {} is unknown. '.format(
                    rule.get("RuleConfigurationName"), job_name
                )
            )
            state["rule_status"] = "Unknown"
            state["next_action"] = "monitor"

    if state["next_action"] == "launch_new" and state['history']['num_retraining'] >= max_num_retraining:
        state["next_action"] = "end"
        stop_job(job_name)
        sns.publish(TopicArn=topic_arn, Message=f'Max number of iterations is reached. Terminating.')

    if state["next_action"] == "launch_new":
        sns.publish(TopicArn=topic_arn, Message=f'Retraining. \n'
                                                f'State: {state}')
        logger.info(f'Retraining. \n'
                    f'State: {state}')

    return {
        'statuscode': 200,
        'body': {
            'state': state
        }
    }

def plan_launch_spec(state):
    """  Read current job params, and prescribe the next training job to launch
    """

    last_run_spec = state['run_spec']
    last_warmup_rate = last_run_spec['warmup_learning_rate']
    add_batch_norm = last_run_spec['add_batch_norm']
    learning_rate = last_run_spec['learning_rate']

    if last_warmup_rate / 5 >= 1e-3:
        logger.info('Reducing warmup rate by 1/5')
        state['history']['num_warmup_adjustments'] += 1
        state['run_spec']['warmup_learning_rate'] = last_warmup_rate * 0.5
        state['next_action'] = 'launch_new'
    elif add_batch_norm == 0:
        logger.info('Adding batch normalization layer')
        state['history']['num_batch_layer_adjustments'] += 1
        state['run_spec']['add_batch_norm'] = 1           # we are only changing the model by adding batch layers
                                                          # prior to ELU. But can make more tweaks here.
        state['next_action'] = 'launch_new'
    elif learning_rate * 0.9 > 0.001:
        state['run_spec']['learning_rate'] = learning_rate * 0.9
        state['history']['num_learning_rate_adjustments'] += 1
        state['next_action'] = 'launch_new'
    else:
        state['next_action'] = 'end'
    return state


def stop_job(job_name):
    """ stop given job"""
    try:
        client.stop_training_job(
            TrainingJobName=job_name
        )
    except client.exceptions.ClientError as e:
        logger.error(
            "Error while attempting to stop job with debugging issue. Job may have finished already. " + str(e)
        )
                        
def stop_processing_job(processing_job_name):
    """ stop given job"""
    try:
        client.stop_processing_job(
            ProcessingJobName=processing_job_name
        )
    except client.exceptions.ClientError as e:
        logger.error(
            "Error while attempting to stop the processing debugger job. Job may have finished already. " + str(e)
        )
