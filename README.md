## Automating complex deep learning model training using Amazon SageMaker Debugger and AWS Step Functions

This repository provides an example workflow to automate model development using AWS Sagemaker Debugger and AWS Step Functions.
[This notebook](resnet-model.ipynb) serves as the entry-point to this sample solution. 

To automatically deploy a Sagemaker Notebook with necessary IAM role and policies to step through the notebook, you 
can use the link to AWS CloudFormation Stack below. Open the link in a new tab to provision the infrastructure required.  Once in 
the AWS CloudFormation page in the AWS Console, accept default entries, review the check box prompt about the creation of IAM roles 
 and policies, and accept if you decide to proceed with deployment of the resources in your AWS account. 

[![CFT](images/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?#/stacks/new?stackName=debugger-cft-stack&templateURL=https://s3.amazonaws.com/aws-ml-blog/artifacts/automating-complex-deep-learning-model-training-SageMaker-Debugger-and-AWS-Step-Functions/debugger_stack.yaml)

Once the stack build has completed, navigate to the [SageMaker Dashboard](https://console.aws.amazon.com/sagemaker/home?/notebook-instances)
 in the AWS Console in your AWS account, and open the newly created "sagemaker-debugger-auto-model-developmentk" notebook. 
 
Step through the notebook to execute the workflow.

### Provisioning resources manually

The AWS CloudFormation stack above provisions everything needed to walk through this solution, including 
the Sagemaker Notebook and the necessary IAM roles and policies. To set up the  IAM roles and policies manually, use the 
policy documents for Sagemaker Notebook and Lambda functions in this [folder](config) for the Sagemaker Notebook execution 
policy and the required policy for AWS Lambda functions. For AWS Step Function role, use the AWS Managed Policy
[`AWSLambdaRole`](https://docs.aws.amazon.com/lambda/latest/dg/access-control-identity-based.html). You can use the AWS 
CloudFormation template [here](config/debugger_stack.yaml) as the blueprint to create the required resources manually.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.