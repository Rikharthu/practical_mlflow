mod model;

use std::fmt::format;
use std::fs::File;
use std::io::Write;
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use clap::{Parser, Subcommand};
use s3::{Bucket, Region};
use s3::creds::Credentials;
use crate::model::{ModelVersionsResponse, find_latest_production_model, ModelManager, ModelVersion, StoredModel};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    mlflow_tracking_uri: String,
    #[arg(long)]
    model_name: String,
}

fn main() {
    let cli = Cli::parse();

    // GET 192.168.0.96:80/api/2.0/mlflow/registered-models/search\
    let uri = format!(
        "{}/api/2.0/mlflow/registered-models/get-latest-versions?name={}",
        cli.mlflow_tracking_uri,
        cli.model_name
    );
    let response = reqwest::blocking::get(uri).unwrap();
    let response = response.json::<ModelVersionsResponse>().unwrap();
    println!("{:?}", response);

    let model = find_latest_production_model(response.model_versions.as_slice())
        .expect("Could not find last production model");
    let model_version = i32::from_str(model.version.as_str()).unwrap();
    println!("Latest model {} {} URI: {}", model.name, model_version, model.source);

    let mut manager = ModelManager::new("models.db");

    let stored_model = manager.get_stored_model(model.name.as_str());
    println!("Currently stored model: {:?}", stored_model);
    if is_model_outdated(model_version, &stored_model) {
        // Model needs to be updated
        println!("Model not found or is outdated, will download latest version from S3");

        let saved_model_path = download_model(&model);
        let updated_model = StoredModel {
            name: model.name.clone(),
            version: model_version,
            path: saved_model_path,
        };

        manager.put_stored_model(&updated_model);

        println!(
            "Saved updated model {} {} to {}",
            updated_model.name,
            updated_model.version,
            updated_model.path
        );
    } else {
        // Model is up to date
        let stored_model = stored_model.unwrap();
        println!("Model {} {} is up to date", stored_model.name, stored_model.version);
    }
}

fn is_model_outdated(model_version: i32, stored_model: &Option<StoredModel>) -> bool {
    stored_model.is_none() || stored_model.as_ref().unwrap().version < model_version
}

fn download_model(model_version: &ModelVersion) -> String {
    let save_path = format!(
        "./models/{}.v{}.pth",
        model_version.name,
        model_version.version
    );

    let model_object_path = format!(
        "{}/{}",
        model_version.source.replace("s3://mlflow", ""),
        "data/model.pth"
    );

    let bucket_name = "mlflow";
    let region = Region::Custom {
        region: "us-east-1".to_string(),
        endpoint: "http://192.168.0.96:9000".to_string(),
    };
    let credentials = Credentials::new(
        Some("minio"),
        Some("minio123"),
        None,
        None,
        None,
    ).unwrap();
    let bucket = Bucket::new(bucket_name, region, credentials).unwrap()
        .with_path_style();

    let data = bucket.get_object_blocking(model_object_path).unwrap();
    let model_bytes = data.bytes();
    let mut model_file = File::create(save_path.as_str()).unwrap();
    model_file.write_all(model_bytes).unwrap();

    save_path
}