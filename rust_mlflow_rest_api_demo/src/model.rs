use std::path::Path;
use std::str::FromStr;
use pickledb::{PickleDb, PickleDbDumpPolicy, SerializationMethod};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelVersionsResponse {
    #[serde(rename = "model_versions")]
    pub model_versions: Vec<ModelVersion>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelVersion {
    pub name: String,
    pub version: String,
    #[serde(rename = "creation_timestamp")]
    pub creation_timestamp: i64,
    #[serde(rename = "last_updated_timestamp")]
    pub last_updated_timestamp: i64,
    #[serde(rename = "current_stage")]
    pub current_stage: String,
    pub description: String,
    pub source: String,
    #[serde(rename = "run_id")]
    pub run_id: String,
    pub status: String,
    #[serde(rename = "run_link")]
    pub run_link: String,
}

pub fn find_latest_production_model(model_version: &[ModelVersion]) -> Option<ModelVersion> {
    model_version.iter()
        .filter(|version| version.current_stage == "Production")
        .max_by(|v1, v2| {
            let v1_version = i32::from_str(v1.version.as_str()).unwrap();
            let v2_version = i32::from_str(v2.version.as_str()).unwrap();
            v1_version.cmp(&v2_version)
        })
        .cloned()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredModel {
    pub name: String,
    pub version: i32,
    pub path: String,
}

pub struct ModelManager {
    db: PickleDb,
}

impl ModelManager {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let mut db = if path.as_ref().exists() {
            PickleDb::load(
                path,
                PickleDbDumpPolicy::AutoDump,
                SerializationMethod::Json,
            ).unwrap()
        } else {
            PickleDb::new(
                path,
                PickleDbDumpPolicy::AutoDump,
                SerializationMethod::Json,
            )
        };

        Self {
            db
        }
    }

    pub fn get_stored_model(&self, name: impl AsRef<str>) -> Option<StoredModel> {
        self.db.get(name.as_ref())
    }

    pub fn put_stored_model(&mut self, model: &StoredModel) {
        self.db.set(model.name.as_str(), model).unwrap();
        self.db.dump().unwrap();
    }
}