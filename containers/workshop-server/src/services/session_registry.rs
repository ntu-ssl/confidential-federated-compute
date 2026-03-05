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

use crate::models::{SessionContext, SessionId};
use crate::launcher_module::Launcher;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

/// Maximum number of concurrent sessions allowed (resource limitation)
const MAX_CONCURRENT_SESSIONS: usize = 60;

/// Struct to hold launcher handles for cleanup
pub struct LauncherHandles {
    pub kms_launcher: Option<Launcher>,
    pub tee_launcher: Option<Launcher>,
}

/// Registry for managing all active sessions
pub struct SessionRegistry {
    sessions: Arc<RwLock<HashMap<SessionId, SessionContext>>>,
    /// Maps session_id to launcher handles for cleanup on session deletion
    launchers: Arc<Mutex<HashMap<SessionId, LauncherHandles>>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        SessionRegistry {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            launchers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new session
    pub async fn create_session(&self, timeout_seconds: u64) -> Result<SessionId> {
        let mut sessions = self.sessions.write().await;

        // Check session limit before creating a new session
        if sessions.len() >= MAX_CONCURRENT_SESSIONS {
            warn!(
                "Session limit reached: {} sessions already active (max: {})",
                sessions.len(),
                MAX_CONCURRENT_SESSIONS
            );
            return Err(anyhow!(
                "Cannot create new session: maximum {} concurrent sessions reached",
                MAX_CONCURRENT_SESSIONS
            ));
        }

        let session_id = SessionId::new();
        let context = SessionContext::new(session_id, timeout_seconds);

        sessions.insert(session_id, context);

        info!("Created session: {} ({}/{} sessions active)", session_id, sessions.len(), MAX_CONCURRENT_SESSIONS);
        Ok(session_id)
    }

    /// Get session context
    pub async fn get_session(&self, session_id: SessionId) -> Result<SessionContext> {
        let sessions = self.sessions.read().await;
        let context = sessions
            .get(&session_id)
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;

        if context.is_expired() {
            warn!("Session {} has expired", session_id);
            return Err(anyhow!("Session {} expired", session_id));
        }

        Ok(context.clone())
    }

    /// Update session context
    pub async fn update_session(&self, session_id: SessionId, context: SessionContext) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions
            .insert(session_id, context)
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;
        Ok(())
    }

    /// Store launcher handles for a session
    pub async fn store_launchers(&self, session_id: SessionId, launchers: LauncherHandles) -> Result<()> {
        let mut launcher_map = self.launchers.lock().await;
        launcher_map.insert(session_id, launchers);
        Ok(())
    }

    /// Get mutable reference to launchers for a session
    /// This is useful for calling methods like get_endorsed_evidence()
    pub async fn get_launchers_mut<F, R>(
        &self,
        session_id: SessionId,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&mut LauncherHandles) -> R,
    {
        let mut launcher_map = self.launchers.lock().await;
        launcher_map.get_mut(&session_id).map(f)
    }

    /// Get and remove launcher handles (for cleanup)
    pub async fn take_launchers(&self, session_id: SessionId) -> Option<LauncherHandles> {
        let mut launcher_map = self.launchers.lock().await;
        launcher_map.remove(&session_id)
    }

    /// Get reference to the launchers map for direct access
    pub fn launchers(&self) -> &Arc<Mutex<HashMap<SessionId, LauncherHandles>>> {
        &self.launchers
    }

    /// Fetch endorsed evidence from the TEE launcher for a session
    /// This method properly handles the async borrow and returns a boxed future
    pub async fn get_tee_endorsed_evidence(
        &self,
        session_id: SessionId,
    ) -> Option<oak_proto_rust::oak::session::v1::EndorsedEvidence> {
        let mut launchers = self.launchers.lock().await;
        if let Some(launcher_handles) = launchers.get_mut(&session_id) {
            if let Some(tee_launcher) = &mut launcher_handles.tee_launcher {
                match tee_launcher.get_endorsed_evidence().await {
                    Ok(evidence) => Some(evidence),
                    Err(_) => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Delete session and kill associated launchers
    pub async fn delete_session(&self, session_id: SessionId) -> Result<()> {
        // Kill launchers first
        if let Some(mut launchers) = self.take_launchers(session_id).await {
            if let Some(mut kms) = launchers.kms_launcher.take() {
                info!("Killing KMS launcher for session: {}", session_id);
                kms.kill().await;
            }
            if let Some(mut tee) = launchers.tee_launcher.take() {
                info!("Killing TEE launcher for session: {}", session_id);
                tee.kill().await;
            }
        }

        // Remove session from registry
        let mut sessions = self.sessions.write().await;
        sessions.remove(&session_id);
        info!("Deleted session: {}", session_id);
        Ok(())
    }

    /// Get current number of active sessions
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    /// Get maximum number of allowed concurrent sessions
    pub fn max_sessions() -> usize {
        MAX_CONCURRENT_SESSIONS
    }

    /// Check if session limit has been reached
    pub async fn is_at_limit(&self) -> bool {
        let sessions = self.sessions.read().await;
        sessions.len() >= MAX_CONCURRENT_SESSIONS
    }

    /// Cleanup expired sessions
    pub async fn cleanup_expired(&self) {
        let mut sessions = self.sessions.write().await;
        let expired: Vec<_> = sessions
            .iter()
            .filter(|(_, ctx)| ctx.is_expired())
            .map(|(id, _)| *id)
            .collect();

        for session_id in expired {
            // Kill launchers for expired sessions
            if let Some(mut launchers) = self.take_launchers(session_id).await {
                if let Some(mut kms) = launchers.kms_launcher.take() {
                    info!("Killing KMS launcher for expired session: {}", session_id);
                    kms.kill().await;
                }
                if let Some(mut tee) = launchers.tee_launcher.take() {
                    info!("Killing TEE launcher for expired session: {}", session_id);
                    tee.kill().await;
                }
            }

            sessions.remove(&session_id);
            info!(
                "Cleaned up expired session: {} ({}/{} remaining)",
                session_id,
                sessions.len(),
                MAX_CONCURRENT_SESSIONS
            );
        }
    }
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SessionRegistry {
    fn clone(&self) -> Self {
        SessionRegistry {
            sessions: Arc::clone(&self.sessions),
            launchers: Arc::clone(&self.launchers),
        }
    }
}
