from google.cloud import storage

bucket_name = "raj_bucket997"
file_name = "Hotel_Reservations.csv"
destination_path = "temp_download.csv"  # Adjust if needed

try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_path)
    print(f"Successfully downloaded to {destination_path}")
except Exception as e:
    print(f"Error downloading: {e}")


    