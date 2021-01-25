import json
import boto3
import logging
import datetime


logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')


def lambda_handler(event, context):
    try:
        state = event['state']
        sm_tensorflow_image = event['sm_tensorflow_image']
        sm_debugger_image = event['sm_debugger_image']
        params = state['run_spec']
         
    except KeyError as e:
        raise KeyError('Key not found in function input.' +
                       'The input received was: {}'.format(json.dumps(event)) + json.dumps(str(e)))

    try:
        job_name = params['base_job_name'] + '-' + datetime.datetime.now().strftime('%Y-%b-%d-%Hh-%Mm-%S')
        
        sm_client.create_training_job(
            TrainingJobName=job_name,
            RoleArn=params['sm_role'],
            AlgorithmSpecification={
                'TrainingImage': sm_tensorflow_image,
                'TrainingInputMode': 'File',
                'EnableSageMakerMetricsTimeSeries': True,
                'MetricDefinitions': [{'Name': 'loss', 'Regex': 'loss: (.+?) '}]
            },
            InputDataConfig=[{'ChannelName': 'train',
                              'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
                                                              'S3Uri': f's3://{params["bucket"]}/data/train',
                                                              'S3DataDistributionType': 'FullyReplicated'}},
                              'CompressionType': 'None',
                              'RecordWrapperType': 'None'},
                             {'ChannelName': 'test',
                              'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
                                                              'S3Uri': f's3://{params["bucket"]}/data/test',
                                                              'S3DataDistributionType': 'FullyReplicated'}},
                              'CompressionType': 'None',
                              'RecordWrapperType': 'None'}],
            HyperParameters={'add_batch_norm': str(params['add_batch_norm']),
                             'model_dir': '"s3://{bucket}/{job_name}/model"'.format(bucket=params['bucket'], job_name=job_name),
                             'num_epochs': str(params['num_epochs']),
                             'sagemaker_container_log_level': '20',
                             'sagemaker_enable_cloudwatch_metrics': 'false',
                             'sagemaker_job_name': json.dumps(job_name),
                             'sagemaker_program': '"train.py"',
                             'sagemaker_region': json.dumps(params['region']),
                             'sagemaker_submit_directory': json.dumps(f"s3://{params['bucket']}/source/sourcedir.tar.gz"),
                             'learning_rate': str(params['learning_rate']),
                             'warmup_learning_rate': str(params['warmup_learning_rate'])},
            OutputDataConfig={'KmsKeyId': '',
                              'S3OutputPath': f's3://{params["bucket"]}'},
            ResourceConfig={'InstanceType': params['instance_type'],
                            'InstanceCount': 1,
                            'VolumeSizeInGB': 30},
            StoppingCondition={'MaxRuntimeInSeconds': 86400},
            DebugHookConfig={'S3OutputPath': f's3://{params["bucket"]}/outputs/debugger',
                             'HookParameters': {'save_interval': '1'},
                             'CollectionConfigurations': [{'CollectionName': 'gradients',
                                                           'CollectionParameters': {'save_interval': str(params["debugger_save_interval"])}}]},
            DebugRuleConfigurations=[{'RuleConfigurationName': 'ExplodingTensor',
                                      'RuleEvaluatorImage': sm_debugger_image,
                                      'VolumeSizeInGB': 0,
                                      'RuleParameters': {'only_nan': 'False',
                                                         'rule_to_invoke': 'ExplodingTensor',
                                                         'tensor_regex': '.*gradient'}}]
        )
        state["history"]['latest_job_name'] = job_name
        state['history']['num_retraining'] = state['history'].get('num_retraining', 0) + 1

    except sm_client.exceptions.ResourceInUse as e:
        return {
            'statusCode': 503,
            'body': json.dumps('Resource not available. ' + str(e))
        }
    except sm_client.exceptions.ResourceLimitExceeded as e:
        return {
            'statusCode': 503,
            'body': json.dumps('Resource limit exceeded. ' + str(e))
        }
    except KeyError as e:
        return {
            'statusCode': 503,
            'body': 'Missing parameter' + json.dumps(str(e))
        }
    except Exception as e:
        return {
            'statusCode': 503,
            'body': json.dumps(str(e))
        }

    return {
        'statusCode': 200,
        'body': {"state": state}
    }
