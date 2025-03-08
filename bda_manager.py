import boto3
import time
import pandas as pd
import json
from urllib.parse import urlparse
import awswrangler as wr


class BedrockDataAutomationManager:
    def __init__(self, region_name='us-east-1'):
        """
        Initializes connections for the BDA service clients.
        """
        self.bda_client = boto3.client('bedrock-data-automation', region_name=region_name)
        self.bda_runtime_client = boto3.client('bedrock-data-automation-runtime', region_name=region_name)
        self.s3_runtime = boto3.client('s3', region_name=region_name)

        self.blueprints = self._list_blueprints()
        self.advertisement_arn = self._get_advertisement_arn()

    def _list_blueprints(self):
        """
        Retrieves a list of BDA blueprints with resourceOwner='SERVICE'.
        """
        response = self.bda_client.list_blueprints(resourceOwner='SERVICE')
        return response["blueprints"]

    def _get_advertisement_arn(self):
        """
        Fetches the ARN of the blueprint with name='Advertisement', if it exists.
        """
        matched = [
            bp for bp in self.blueprints
            if bp.get("blueprintName") == "Advertisement"
        ]
        return matched[0].get("blueprintArn") if matched else None

    def _project_exists(self, project_name: str) -> bool:
        """
        Returns True if a Data Automation project with the given name already exists.
        """
        # list_data_automation_projects supports pagination;
        # for simplicity we collect all results in one list.
        paginator = self.bda_client.get_paginator('list_data_automation_projects')
        for page in paginator.paginate():
            for project in page.get("projects", []):
                if project.get("projectName") == project_name:
                    return project
        return False

    def create_bda_project(self, *,
                           project_name: str,
                           project_description: str,
                           project_stage: str):
        """
        Creates a new Data Automation project with given parameters,
        unless a project of the same name already exists.
        """
        __project_exist = self._project_exists(project_name)
        if __project_exist:
            print(f"Skipping creation: a project named '{project_name}' already exists.")
            return __project_exist

        if not self.advertisement_arn:
            raise ValueError("Advertisement blueprint ARN not found.")

        response_bda_creation = self.bda_client.create_data_automation_project(
            projectName=project_name,
            projectDescription=project_description,
            projectStage=project_stage,
            standardOutputConfiguration={
                'document': {
                    'extraction': {
                        'granularity': {
                            'types': ['PAGE', 'ELEMENT']
                        },
                        'boundingBox': {'state': 'ENABLED'}
                    },
                    'generativeField': {'state': 'ENABLED'},
                    'outputFormat': {
                        'textFormat': {'types': ['MARKDOWN']},
                        'additionalFileFormat': {'state': 'ENABLED'}
                    }
                },
                'image': {
                    'extraction': {
                        'category': {
                            'state': 'ENABLED',
                            'types': ['TEXT_DETECTION']
                        },
                        'boundingBox': {'state': 'ENABLED'}
                    },
                    'generativeField': {
                        'state': 'ENABLED',
                        'types': ['IMAGE_SUMMARY']
                    }
                },
                'video': {
                    'extraction': {
                        'category': {
                            'state': 'ENABLED',
                            'types': ['TRANSCRIPT']
                        },
                        'boundingBox': {'state': 'ENABLED'}
                    },
                    'generativeField': {
                        'state': 'ENABLED',
                        'types': ['VIDEO_SUMMARY']
                    }
                },
                'audio': {
                    'extraction': {
                        'category': {
                            'state': 'ENABLED',
                            'types': ['TRANSCRIPT']
                        }
                    },
                    'generativeField': {
                        'state': 'ENABLED',
                        'types': ['AUDIO_SUMMARY']
                    }
                }
            },
            customOutputConfiguration={
                'blueprints': [
                    {
                        'blueprintArn': self.advertisement_arn,
                        'blueprintStage': 'LIVE'
                    },
                ]
            },
        )
        return response_bda_creation

    def get_bda_project(self, *,
                        project_arn: str,
                        project_stage: str):
        """
        Retrieves a Data Automation project by ARN and stage.
        """
        return self.bda_client.get_data_automation_project(
            projectArn=project_arn,
            projectStage=project_stage
        )

    def run_bda_inference(self, *,
                          bda_arn: str,
                          input_s3_uri: str,
                          output_s3_uri: str,
                          data_automation_profile_arn: str):
        """
        Invokes a BDA inference job asynchronously.
        """
        response_bda_run = self.bda_runtime_client.invoke_data_automation_async(
            inputConfiguration={'s3Uri': input_s3_uri},
            outputConfiguration={'s3Uri': output_s3_uri},
            dataAutomationConfiguration={
                'dataAutomationProjectArn': bda_arn,
                'stage': 'LIVE'
            },
            dataAutomationProfileArn=data_automation_profile_arn
        )
        return response_bda_run

    def get_bda_inference_status(self, *,
                                 invocation_arn: str):
        """
        Gets the status of a BDA inference job.
        """
        return self.bda_runtime_client.get_data_automation_status(
            invocationArn=invocation_arn
        )

    def read_custom_out_path_and_load(self, *, final_df: pd.DataFrame,
                                      path_column: str = 'custom_output_path') -> pd.DataFrame:
        """
        For each element in 'path_column' of final_df, read the JSON file from S3 and combine all data into one DataFrame.
        """
        dfs = []

        for idx, s3_uri in final_df[path_column].dropna().items():
            try:
                df_temp = wr.s3.read_json(path=s3_uri, orient='records', lines=True)
                # Optionally append row index or any other metadata
                df_temp['source_index'] = idx
                df_temp['source_file'] = s3_uri
                dfs.append(df_temp)
            except Exception as e:
                print(f"Error reading from {s3_uri}: {e}")

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def retrieve_inference_results(self, *,
                                   s3_uri: str):
        """
        Retrieves and processes the inference results from the specified S3 URI.
        Returns a Pandas DataFrame containing the output data.
        """

        df_invocation_metadata = wr.s3.read_json(path=s3_uri)

        df_invocation_metadata = pd.concat(
            [df_invocation_metadata, pd.json_normalize(df_invocation_metadata['output_metadata'])], axis=1)

        # Combine metadata with normalized versions of nested fields
        df_exploded_output = df_invocation_metadata.explode('segment_metadata').reset_index(drop=True)

        return pd.concat(
            [df_exploded_output, pd.json_normalize(df_exploded_output['segment_metadata'])], axis=1)


def main():
    # Example usage
    # get account account id though sts
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    project_name = 'BedrockDataAutomationProject'
    project_description = 'My E-Commerce Bedrock Data Automation (BDA)'
    project_stage = 'LIVE'
    bda_bucket_name = "bedrock-bda-us-east-1-31f08eea-50d4-44d9-8872-92a1dd69e4bd"
    # input_s3_uri = 's3://bedrock-bda-us-east-1-31f08eea-50d4-44d9-8872-92a1dd69e4bd/input/burgerad15179945781952220614.png'
    input_s3_uri = 's3://bedrock-bda-us-east-1-31f08eea-50d4-44d9-8872-92a1dd69e4bd/input/teamsConversation.mp4'
    output_s3_uri = 's3://bedrock-bda-us-east-1-31f08eea-50d4-44d9-8872-92a1dd69e4bd/inference_results'
    data_automation_profile_arn = f'arn:aws:bedrock:us-east-1:{account_id}:data-automation-profile/us.data-automation-v1'

    # Initialize manager
    manager = BedrockDataAutomationManager()

    # Create a new BDA project
    response_bda_creation = manager.create_bda_project(
        project_name=project_name,
        project_description=project_description,
        project_stage=project_stage
    )
    print("Project creation response:")
    print(response_bda_creation)

    # Get project status
    project_arn = response_bda_creation["projectArn"]
    project_stage = response_bda_creation["projectStage"]
    response_bda_status = manager.get_bda_project(
        project_arn=project_arn,
        project_stage=project_stage
    )
    print("Project status:")
    print(response_bda_status)

    # Invoke inference
    response_bda_run = manager.run_bda_inference(
        bda_arn=response_bda_status["project"]["projectArn"],
        input_s3_uri=input_s3_uri,
        output_s3_uri=output_s3_uri,
        data_automation_profile_arn=data_automation_profile_arn
    )

    # Initial status retrieval
    response_bda_run_status = manager.get_bda_inference_status(
        invocation_arn=response_bda_run["invocationArn"]
    )

    # Loop until the status is no longer "InProgress"
    while response_bda_run_status.get('status') == "InProgress":
        print("Inference is still in progress. Waiting for completion...")
        time.sleep(10)  # Wait for 10 seconds before checking again
        response_bda_run_status = manager.get_bda_inference_status(
            invocation_arn=response_bda_run["invocationArn"]
        )

    print("Inference completed with status:", response_bda_run_status.get('status'))
    print("Inference status:")
    print(response_bda_run_status)
    invocation_id = response_bda_run['invocationArn'].split("/")[1]
    # Retrieve final results
    output_conf = response_bda_run_status.get("outputConfiguration")
    if output_conf and "s3Uri" in output_conf:
        exploded_job_metadata = manager.retrieve_inference_results(s3_uri=output_conf["s3Uri"])
        print("Final DataFrame:")
        print(exploded_job_metadata)
        modality_extracted = exploded_job_metadata["semantic_modality"][0]
        # check if "custom_output_path" in columns of exploded_job_metadata
        custom_extracted_details = pd.DataFrame()
        if "custom_output_path" in exploded_job_metadata.columns:
            custom_extracted_details = manager.read_custom_out_path_and_load(final_df=exploded_job_metadata,
                                                                             path_column="custom_output_path")

            custom_extracted_details["blueprintName"] = pd.json_normalize(
                custom_extracted_details["matched_blueprint"]
            )["name"]

        standard_extracted_details = manager.read_custom_out_path_and_load(final_df=exploded_job_metadata,
                                                                           path_column="standard_output_path")
        # loop through the different modalities available in the BDA projects either "image", "video"
        if modality_extracted == "IMAGE":
            __normalized_image_details = pd.json_normalize(standard_extracted_details["image"])

            standard_extracted_details["summary"] = __normalized_image_details["summary"]
            standard_extracted_details["extracted_text_words"] = __normalized_image_details["text_words"]
            standard_extracted_details["extracted_text_lines"] = __normalized_image_details["text_lines"]
        elif modality_extracted == "VIDEO":
            __normalized_image_details = pd.json_normalize(standard_extracted_details["video"])

            standard_extracted_details["summary"] = __normalized_image_details["summary"]
            standard_extracted_details["extracted_transcript"] = __normalized_image_details[
                "transcript.representation.text"]

        final_result_bda_pd = pd.concat(
            [
                custom_extracted_details,
                standard_extracted_details
            ]
        )
        # save final_result_bda_pd to S3
        wr.s3.to_parquet(
            df=final_result_bda_pd,
            path=f"s3://{bda_bucket_name}/output/{project_name}/{invocation_id}/final_result_bda_pd.parquet",
            index=False
        )



if __name__ == "__main__":
    main()
