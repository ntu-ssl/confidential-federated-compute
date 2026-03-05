// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod handlers;
pub mod routes;

use crate::services::{LauncherManager, SessionRegistry};
use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use tower_http::trace::TraceLayer;

/// Create the Axum router with all configured routes
pub fn create_router(
    registry: Arc<SessionRegistry>,
    launcher_manager: Arc<LauncherManager>,
) -> Router {
    Router::new()
        .route("/health", get(handlers::get_status))
        .route("/status", get(handlers::get_status))
        .route("/sessions", post(handlers::create_session))
        .route("/sessions/:session_id", get(handlers::get_session))
        .route("/sessions/:session_id", delete(handlers::delete_session))
        .route("/sessions/:session_id/kms/start", post(handlers::start_kms))
        .route("/sessions/:session_id/tee/start", post(handlers::start_tee))
        .route(
            "/sessions/:session_id/tee/:instance_id/evidence",
            get(handlers::get_tee_evidence),
        )
        .route(
            "/sessions/:session_id/tee/:instance_id/reference-values",
            get(handlers::get_tee_reference_values),
        )
        .route("/sessions/:session_id/kms/rotate-keyset", post(handlers::rotate_keyset))
        .route("/sessions/:session_id/kms/derive-keys", post(handlers::derive_keys))
        .route(
            "/sessions/:session_id/kms/register-pipeline",
            post(handlers::register_pipeline),
        )
        .route(
            "/sessions/:session_id/kms/authorize-transform",
            post(handlers::authorize_transform),
        )
        .route(
            "/sessions/:session_id/variant-policies",
            post(handlers::create_variant_policy),
        )
        .route(
            "/sessions/:session_id/data-access-policies",
            post(handlers::create_data_access_policy),
        )
        .route(
            "/sessions/:session_id/encrypt-data",
            post(handlers::encrypt_data),
        )
        .route(
            "/sessions/:session_id/tee/process",
            post(handlers::process_with_tee),
        )
        .with_state((registry, launcher_manager))
        .layer(TraceLayer::new_for_http())
}
