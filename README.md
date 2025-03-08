Overview:
---------
The “bda_manager.py” script demonstrates how to set up, manage, and run AWS Bedrock Data Automation (BDA) projects. It includes functionality to create a new project, invoke inference on video or image data, retrieve resulting metadata, and process those results. In this example, I showcase the “advertisement” pre-made blueprint, illustrating how you can quickly incorporate various ready-to-use blueprints in BDA (There are various pre-made blueprints, such as payslip processing, moderation, ...)

Main Functionalities:
---------------------
1. Creating a Bedrock Data Automation project:
   - Uses the AWS Bedrock Data Automation APIs to set up a project with a specified name, description, and stage.
   - Returns the project’s ARN (Amazon Resource Name) and status information.

2. Retrieving existing BDA project details:
   - Given an existing project’s ARN and stage, the script retrieves and prints the project’s status.

3. Invoking a BDA inference job:
   - Submits data (image or video) to an existing BDA project for inference.
   - S3 paths for input and output are provided, along with the data automation profile ARN specifying inference parameters.

4. Retrieving and processing BDA output:
   - Reads the output JSON from S3, normalizes nested metadata fields, and merges them into a Pandas DataFrame.
   - Uses the “explode” function to handle a list of segment metadata returned from BDA and flattens it out into separate rows.
   - Optionally merges together the standard_output_path and custom_output_path metadata into a 360 view.

Usage Instructions:
------------------
1. Ensure you have Python 3.7+ installed.
2. Install required dependencies:
   pip install boto3 awswrangler pandas

3. Provide valid AWS credentials and region in your environment (e.g., using AWS CLI, environment variables, credentials file, etc.).
4. Update the main() function parameters with your own: 
   - Project name and description
   - Input S3 URI (e.g., an image or video file)
   - Output S3 URI
   - Data automation profile ARN  
5. Run the script:
   python bda_manager.py

6. Once run, the script will: 
   - Create or retrieve the specified BDA project
   - Trigger the data automation pipeline
   - Periodically check and print status updates
   - Upon completion, retrieve S3 output, process results, and store them back in an S3 location for your reference


Customization & Extensibility:
------------------------------
• Extending the script for different or additional BDA Blueprints:
  - Add or modify blueprint configuration details in the create_bda_project method.  
• Handling different input data types:
  - The script is currently geared toward IMAGE and VIDEO data. You can add more logic for other media or text-based data if supported by your BDA project.
• Merging additional output fields:
  - Extend retrieve_inference_results function or read_custom_out_path_and_load to parse and flatten other nested JSON structures.
