mod model;

use std::fs::File;
use std::io::Write;
use serde::{Deserialize, Serialize};
use clap::{Parser, Subcommand};
use s3::{Bucket, Region};
use s3::creds::Credentials;
use crate::model::{ModelVersionsResponse, find_latest_production_model};

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
    println!("Latest model {} {} URI: {}", model.name, model.version, model.source);

    let model_object_path = format!(
        "{}/{}",
        model.source.replace("s3://mlflow", ""),
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
    // TODO: use returned URI
    let data = bucket.get_object_blocking(model_object_path).unwrap();
    let model_bytes = data.bytes();
    let mut model_file = File::create("./model.pth").unwrap();
    model_file.write_all(model_bytes).unwrap();
}
