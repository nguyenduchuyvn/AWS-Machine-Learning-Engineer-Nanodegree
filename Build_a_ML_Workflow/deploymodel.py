import json
import boto3
import base64


# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-07-13-06-56-23-058"

runtime_client = boto3.Session().client('sagemaker-runtime')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Decode the image data
    image = base64.b64decode(event["body"]['image_data'])

    response = runtime_client.invoke_endpoint(EndpointName=ENDPOINT,
                                                Body=image,
                                                ContentType='image/png')
    
    # For this model the IdentitySerializer needs to be "image/png"
    # predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = json.loads(response['Body'].read().decode('utf-8'))     # list
    
    # We return the data back to the Step Function 
    # event['inferences'] = inferences
    

    return {
        'statusCode': 200,
        'inferences': inferences
    }