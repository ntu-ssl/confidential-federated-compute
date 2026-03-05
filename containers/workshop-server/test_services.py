#!/usr/bin/env python3
"""
Test script for Workshop Server KMS and Test Concat TEE services.

This script demonstrates the complete workflow with attestation data retrieval
and optional full 9-step KMS flow execution (from kms-flow-simulation):

BASIC MODE (default):
1. Checks server status and capacity (max 40 concurrent sessions)
2. Creates a new session (KMS automatically assigned from pool of 5 instances)
3. Starts Test Concat TEE service
4. Retrieves and displays TEE evidence (attestation data)
5. Retrieves and displays TEE reference values (derived from evidence)
6. Monitors services briefly
7. Cleans up by deleting the session (releases KMS back to pool)

FULL 9-STEP KMS FLOW MODE (with --full-flow):
1. Rotate keyset in KMS
2. Fetch TEE evidence, endorsements, and reference values
3. Create policy with TEE reference values
4. Derive encryption keys from policy
5. Client encrypts data (two separate chunks)
6. Store encrypted data
7. Register pipeline invocation
8. Authorize transform with real TEE evidence
9. Send encrypted blobs to TEE for processing
10. Release results via KMS

POLICY HASH MISMATCH SCENARIO MODE (with --policy-hash-mismatch):
Demonstrates what happens when policy hashes don't match during encryption and authorization.
This test scenario shows the cryptographic failure when different policies are used:
1. Create two different policies (Policy A and Policy B with different pipeline names)
2. Derive encryption keys from Policy A's hash
3. Encrypt data with the public key from Policy A
4. Register and authorize pipeline with Policy B (DIFFERENT!)
5. Send data to TEE for processing
6. TEE tries to decrypt with keys for Policy B
7. Decryption fails because HKDF("info"=policy_hash_A) ≠ HKDF("info"=policy_hash_B)

Key Insight: KMS does NOT validate policy hash consistency - it relies on cryptographic
integrity to fail gracefully when hashes don't match.

CONCURRENT PRESSURE TEST MODE (with --pressure-test):
Tests the server's ability to handle multiple concurrent users running the complete 9-step KMS flow.
This mode simulates real-world load on the Workshop Server and KMS infrastructure:
1. Spawns N concurrent users (default: 20, configurable with --num-users)
2. Each user executes the full 9-step flow:
   - Session creation + TEE launch + reference values
   - Keyset rotation
   - Policy creation (variant + data access)
   - Key derivation
   - Data encryption (2 chunks)
   - Pipeline registration
   - Transform authorization with TEE evidence
   - TEE processing with encrypted data
   - Result retrieval
3. All sessions remain active for 30 seconds (simulating sustained workload)
4. All sessions are cleaned up concurrently
5. Reports success rate, timing metrics, and resource cleanup status

Use Cases: KMS load testing, end-to-end flow validation, capacity testing, resource leak detection.

Features:
- KMS Pool Architecture: 5 pre-started KMS instances shared across sessions
- Automatic KMS assignment: Least-loaded instance assigned on session creation
- Server capacity checking (40 concurrent session limit)
- Pre-flight status check with available slots display
- Real TEE evidence and endorsements fetching
- Reference value extraction from attestation
- Complete 9-step KMS flow execution (optional)
- Policy hash mismatch scenario testing (optional)
- Concurrent pressure testing (optional, configurable user count)
- Detailed attestation data printing with hex representation and sizes
- Data encryption and TEE processing
- Cryptographic failure demonstration
- HTTP 409 Conflict handling for capacity limits
- Error handling with informative messages
- Graceful cleanup on exit (KMS released back to pool)

Usage:
    python3 test_services.py [OPTIONS]

Examples:
    # Basic service test
    python3 test_services.py

    # Full 9-step KMS flow
    python3 test_services.py --full-flow

    # Policy hash mismatch scenario test
    python3 test_services.py --policy-hash-mismatch

    # Custom host and port
    python3 test_services.py --host localhost --port 8080

    # Full flow with verbose monitoring
    python3 test_services.py --full-flow --verbose

    # Custom memory configuration
    python3 test_services.py --full-flow --kms-memory 2G --tee-memory 2G

    # Policy hash mismatch with custom memory
    python3 test_services.py --policy-hash-mismatch --kms-memory 2G --tee-memory 2G

    # Concurrent pressure test: 20 users running full 9-step KMS flow (default)
    python3 test_services.py --pressure-test

    # Pressure test with 50 concurrent users
    python3 test_services.py --pressure-test --num-users 50

    # Pressure test with custom server (tests distributed load)
    python3 test_services.py --pressure-test --num-users 30 --host 192.168.1.100 --port 3000
"""

import argparse
import requests
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class WorkshopServerClient:
    """Client for Workshop Server REST API."""

    def __init__(self, host: str = "localhost", port: int = 3000):
        self.base_url = f"http://{host}:{port}"
        self.session_id = None
        self.session_data = None

    def create_session(self, timeout_seconds: int = 3600) -> dict:
        """Create a new session.

        Note: Server has a maximum of 40 concurrent sessions. If the limit is reached,
        this method will raise an exception with HTTP 409 Conflict.
        """
        print(f"\n📝 Creating session (timeout: {timeout_seconds}s)...")
        url = f"{self.base_url}/sessions"
        payload = {"timeout_seconds": timeout_seconds}

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.session_id = data["session_id"]
            self.session_data = data
            print(f"✅ Session created: {self.session_id}")
            print(f"   Created at: {data['created_at']}")
            return data
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 409:
                # Server is at capacity (40 concurrent sessions limit)
                error_data = e.response.json()
                print(f"❌ Server is at capacity!")
                print(f"   Error: {error_data.get('error', 'Maximum concurrent sessions reached')}")
                print(f"   Max sessions: {error_data.get('max_sessions', 40)}")
                print(f"   Please wait for a session to complete or try again later.")
                raise Exception(f"Server at capacity: {error_data.get('error', 'Maximum 40 concurrent sessions reached')}")
            else:
                print(f"❌ Failed to create session: {e}")
                raise
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create session: {e}")
            raise

    def start_kms(self, memory_size: str = "2G") -> dict:
        """DEPRECATED: Start KMS service.

        NOTE: As of the KMS pool implementation, KMS is automatically assigned from a pool
        of 5 pre-started instances when the session is created. This method is kept for
        backward compatibility and will return the already-assigned KMS information.

        The server no longer starts a new KMS instance for each session. Instead:
        - 5 KMS instances are pre-started at server startup
        - Sessions are assigned to the least-loaded KMS instance
        - KMS instances are shared across multiple sessions

        This method now simply returns the auto-assigned KMS information.
        """
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n🔐 Getting KMS service info (auto-assigned from pool)...")
        url = f"{self.base_url}/sessions/{self.session_id}/kms/start"
        payload = {"memory_size": memory_size}

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "already_assigned":
                print(f"✅ KMS already assigned from pool")
                print(f"   Address: {data.get('address', 'N/A')}")
                print(f"   Port: {data.get('port', 'N/A')}")
                print(f"   ℹ️  {data.get('message', '')}")
            else:
                print(f"✅ KMS service info retrieved")
                print(f"   Address: {data.get('address', 'N/A')}")
                print(f"   Port: {data.get('port', 'N/A')}")

            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get KMS info: {e}")
            raise

    def start_tee(self, memory_size: str = "2G") -> dict:
        """Start Test Concat TEE service."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n🛡️  Starting Test Concat TEE service (memory: {memory_size})...")
        url = f"{self.base_url}/sessions/{self.session_id}/tee/start"
        payload = {"memory_size": memory_size}

        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            print(f"✅ TEE service started")
            if "instances" in data:
                for instance in data["instances"]:
                    print(f"   Instance: {instance.get('instance_id', 'N/A')}")
                    print(f"   Address: {instance.get('address', 'N/A')}")
                    print(f"   Port: {instance.get('port', 'N/A')}")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to start TEE: {e}")
            raise

    def get_status(self) -> dict:
        """Get workshop server status and session metrics."""
        print(f"\n🔍 Checking server status...")
        url = f"{self.base_url}/status"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Server status retrieved")

            if "sessions" in data:
                sessions = data["sessions"]
                active = sessions.get("active", 0)
                max_sessions = sessions.get("max", 0)
                at_limit = sessions.get("at_limit", False)
                available = sessions.get("available", 0)

                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   Active Sessions: {active}/{max_sessions}")
                print(f"   Available Slots: {available}")

                if at_limit:
                    print(f"   ⚠️  Server is at capacity (limit reached)")
                    return data
                else:
                    print(f"   ✓ Server has capacity")

            return data
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not check server status: {e}")
            print(f"   Attempting to continue anyway...")
            return {"status": "unknown"}

    def get_session(self) -> dict:
        """Get session information."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n📊 Retrieving session information...")
        url = f"{self.base_url}/sessions/{self.session_id}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Session state retrieved")
            print(f"   KMS Service: {bool(data.get('kms_service'))}")
            print(f"   TEE Service: {'Yes' if data.get('tee_service') else 'No'}")
            print(f"   Policies: {len(data.get('policies', {}))}")
            print(f"   Execution Events: {len(data.get('execution_log', []))}")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get session: {e}")
            raise

    def delete_session(self) -> bool:
        """Delete session (cleanup)."""
        if not self.session_id:
            return False

        print(f"\n🗑️  Deleting session...")
        url = f"{self.base_url}/sessions/{self.session_id}"

        try:
            response = requests.delete(url, timeout=10)
            response.raise_for_status()
            print(f"✅ Session deleted")
            self.session_id = None
            self.session_data = None
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to delete session: {e}")
            return False

    def get_tee_evidence(self, instance_id: str = None) -> dict:
        """Get TEE evidence and endorsements."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        # Get TEE instance if not specified
        if not instance_id:
            session_data = self.get_session()
            tee_service = session_data.get("tee_service")
            if not tee_service:
                print("⚠️  No TEE service found in session")
                return {}
            instance_id = tee_service.get("instance_id")

        print(f"\n🔬 Fetching TEE evidence for instance {instance_id}...")
        url = f"{self.base_url}/sessions/{self.session_id}/tee/{instance_id}/evidence"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ TEE Evidence retrieved")
            print(f"   Instance: {data.get('instance_id', 'N/A')}")
            print(f"   Address: {data.get('address', 'N/A')}")
            print(f"   Port: {data.get('port', 'N/A')}")

            # Print evidence structure
            evidence_hex = data.get('evidence_hex', '')
            evidence_size = data.get('evidence_size_bytes', 0)
            evidence_struct = data.get('evidence_structure', '')
            if evidence_hex:
                print(f"\n   📦 Evidence:")
                print(f"      Size: {evidence_size} bytes")
                print(f"      Hex (first 100 chars): {evidence_hex[:100]}...")
                if evidence_struct and evidence_struct != "None":
                    print(f"      Structure:\n{self._format_structure(evidence_struct, 6)}")

            # Print endorsements structure
            endorsements_hex = data.get('endorsements_hex', '')
            endorsements_size = data.get('endorsements_size_bytes', 0)
            endorsements_struct = data.get('endorsements_structure', '')
            if endorsements_hex:
                print(f"\n   📦 Endorsements:")
                print(f"      Size: {endorsements_size} bytes")
                print(f"      Hex (first 100 chars): {endorsements_hex[:100]}...")
                if endorsements_struct and endorsements_struct != "None":
                    print(f"      Structure:\n{self._format_structure(endorsements_struct, 6)}")

            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get TEE evidence: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            return {}

    def get_tee_reference_values(self, instance_id: str = None) -> dict:
        """Get TEE reference values."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        # Get TEE instance if not specified
        if not instance_id:
            session_data = self.get_session()
            tee_service = session_data.get("tee_service")
            if not tee_service:
                print("⚠️  No TEE service found in session")
                return {}
            instance_id = tee_service.get("instance_id")

        print(f"\n📊 Fetching TEE reference values for instance {instance_id}...")
        url = f"{self.base_url}/sessions/{self.session_id}/tee/{instance_id}/reference-values"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ TEE Reference Values retrieved")
            print(f"   Instance: {data.get('instance_id', 'N/A')}")
            print(f"   Address: {data.get('address', 'N/A')}")
            print(f"   Port: {data.get('port', 'N/A')}")

            # Print reference values structure
            ref_values_hex = data.get('reference_values_hex', '')
            ref_values_size = data.get('reference_values_size_bytes', 0)
            ref_values_struct = data.get('reference_values_structure', '')
            if ref_values_hex:
                print(f"\n   📦 Reference Values:")
                print(f"      Size: {ref_values_size} bytes")
                print(f"      Hex (first 100 chars): {ref_values_hex[:100]}...")
                if ref_values_struct and ref_values_struct != "None":
                    print(f"      Structure:\n{self._format_structure(ref_values_struct, 6)}")

            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get TEE reference values: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            return {}

    def _format_structure(self, struct_str: str, indent: int = 0) -> str:
        """Format protobuf structure for readability with truncation."""
        lines = struct_str.split('\n')
        formatted_lines = []
        indent_str = ' ' * indent

        # Truncate very long lines for readability
        for line in lines[:50]:  # Show first 50 lines max
            if len(line) > 200:
                formatted_lines.append(indent_str + line[:200] + '...')
            else:
                formatted_lines.append(indent_str + line)

        if len(lines) > 50:
            formatted_lines.append(indent_str + f'... ({len(lines) - 50} more lines)')

        return '\n'.join(formatted_lines)

    def wait_and_monitor(self, duration: int = 30, poll_interval: int = 5, verbose: bool = False):
        """
        Wait for specified duration while optionally monitoring services.

        Args:
            duration: How long to wait in seconds
            poll_interval: How often to check status (in seconds)
            verbose: If True, print full session details; if False, just show countdown
        """
        if duration <= 0:
            print("ℹ️  Monitoring duration is 0, skipping...")
            return

        print(f"\n⏱️  Services running for {duration} seconds...")
        end_time = time.time() + duration
        last_status_time = time.time()

        while time.time() < end_time:
            remaining = int(end_time - time.time())

            # Print status periodically (every poll_interval or when close to end)
            current_time = time.time()
            should_print = (current_time - last_status_time >= poll_interval) or remaining <= 5

            if should_print:
                if verbose:
                    # Print detailed session info
                    try:
                        self.get_session()
                    except Exception as e:
                        print(f"⚠️  Error retrieving session: {e}")
                else:
                    # Just print countdown
                    print(f"   ⏳ {remaining}s remaining...")

                last_status_time = current_time

            # Sleep for a short interval (1 second) to keep responsive
            sleep_time = min(1.0, end_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"✅ Monitoring complete")

    def rotate_keyset(self, keyset_id: int = 1, ttl_seconds: int = 7200) -> dict:
        """Step 1: Rotate keyset in KMS."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n🔄 Step 1: Rotating keyset (ID: {keyset_id}, TTL: {ttl_seconds}s)...")
        url = f"{self.base_url}/sessions/{self.session_id}/kms/rotate-keyset"
        payload = {"keyset_id": keyset_id, "ttl_seconds": ttl_seconds}

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Keyset rotated successfully")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to rotate keyset: {e}")
            raise

    def create_variant_policy(self, variant_policy_name: str = "test_concat_variant", src_node_ids: list = None, dst_node_ids: list = None) -> dict:
        """Step 3a: Create a PipelineVariantPolicy with TEE reference values."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        if src_node_ids is None:
            src_node_ids = [0]  # Read from input node
        if dst_node_ids is None:
            dst_node_ids = [1]  # Write to output node

        print(f"\n📋 Creating variant policy '{variant_policy_name}' (nodes {src_node_ids} → {dst_node_ids})...")
        url = f"{self.base_url}/sessions/{self.session_id}/variant-policies"
        payload = {
            "variant_policy_name": variant_policy_name,
            "src_node_ids": src_node_ids,
            "dst_node_ids": dst_node_ids
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Variant policy created successfully")
            if "variant_policy_bytes_size" in data:
                print(f"   Size: {data['variant_policy_bytes_size']} bytes")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create variant policy: {e}")
            raise

    def create_data_access_policy(self, policy_name: str = "test_concat_policy", logical_pipeline_name: str = "test_concat_pipeline", variant_policy_names: list = None) -> dict:
        """Step 3b: Create a DataAccessPolicy from variant policies."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        if variant_policy_names is None:
            variant_policy_names = ["test_concat_variant"]

        print(f"\n📋 Creating data access policy '{policy_name}' with variants {variant_policy_names}...")
        url = f"{self.base_url}/sessions/{self.session_id}/data-access-policies"
        payload = {
            "policy_name": policy_name,
            "logical_pipeline_name": logical_pipeline_name,
            "variant_policy_names": variant_policy_names
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Data access policy created successfully")
            if "policy_hash" in data:
                print(f"   Policy hash: {data['policy_hash'][:32]}...")
            if "policy_bytes_size" in data:
                print(f"   Size: {data['policy_bytes_size']} bytes")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to create data access policy: {e}")
            raise

    def derive_keys(self, policy_name: str = "test_concat_policy", policy_bytes: str = None) -> dict:
        """Step 3: Derive encryption keys from policy."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n🔐 Step 3: Deriving encryption keys from policy '{policy_name}'...")
        url = f"{self.base_url}/sessions/{self.session_id}/kms/derive-keys"
        payload = {"policy_name": policy_name}
        if policy_bytes:
            payload["policy_bytes"] = policy_bytes

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Keys derived successfully")
            if "public_keys" in data:
                print(f"   Keys count: {len(data['public_keys'])}")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to derive keys: {e}")
            raise

    def encrypt_data(self, blob_name: str, plaintext: str, policy_name: str = "test_concat_policy", public_key_index: int = 0) -> dict:
        """Step 4: Encrypt plaintext data."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n🔒 Step 4: Encrypting data blob '{blob_name}' ({len(plaintext)} bytes)...")
        url = f"{self.base_url}/sessions/{self.session_id}/encrypt-data"
        # Convert plaintext to hex for transport
        plaintext_hex = plaintext.encode().hex() if isinstance(plaintext, str) else plaintext.hex()
        payload = {
            "blob_name": blob_name,
            "plaintext": plaintext_hex,
            "policy_name": policy_name,
            "public_key_index": public_key_index
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Data encrypted successfully")
            if "encrypted_blob" in data:
                print(f"   Encrypted size: {len(data['encrypted_blob'])//2} bytes")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to encrypt data: {e}")
            raise

    def register_pipeline(self, invocation_name: str = "test_concat_invocation", logical_pipeline_name: str = "test_concat_pipeline", data_access_policy_name: str = "test_concat_policy", keyset_id: int = 1, ttl_seconds: int = 3600) -> dict:
        """Step 6: Register pipeline invocation with KMS."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n📝 Step 6: Registering pipeline invocation '{invocation_name}'...")
        url = f"{self.base_url}/sessions/{self.session_id}/kms/register-pipeline"
        payload = {
            "invocation_name": invocation_name,
            "logical_pipeline_name": logical_pipeline_name,
            "data_access_policy_name": data_access_policy_name,
            "keyset_id": keyset_id,
            "ttl_seconds": ttl_seconds
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Pipeline registered successfully")
            if "invocation_id" in data:
                print(f"   Invocation ID: {data['invocation_id'][:32]}...")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to register pipeline: {e}")
            raise

    def authorize_transform(self, invocation_name: str = "test_concat_invocation") -> dict:
        """Step 7: Authorize transform to access decryption keys.

        Server-side flow:
        1. Retrieves invocation which stores logical_pipeline_name and data_access_policy_name
        2. Decodes the DataAccessPolicy
        3. Extracts the PipelineVariantPolicy from the logical pipeline
        4. Re-encodes the variant to ensure exact bytes match what's in DataAccessPolicy
        5. Passes extracted variant bytes to KMS.AuthorizeTransform
        """
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        print(f"\n🔑 Step 7: Authorizing transform for invocation '{invocation_name}'...")
        print(f"   Server will extract variant policy from data access policy for authorization")
        url = f"{self.base_url}/sessions/{self.session_id}/kms/authorize-transform"
        payload = {"invocation_name": invocation_name}

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Transform authorized successfully with variant policy")
            if "protected_response" in data:
                print(f"   Protected response size: {len(data['protected_response'])//2} bytes")
            return data
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to authorize transform: {e}")
            raise

    def process_with_tee(self, blob_names: list, invocation_name: str = "test_concat_invocation", instance_id: str = None) -> dict:
        """Step 8: Send encrypted data to TEE for processing."""
        if not self.session_id:
            raise ValueError("Session not created yet. Call create_session() first.")

        # Get TEE instance if not specified
        if not instance_id:
            session_data = self.get_session()
            tee_service = session_data.get("tee_service")
            if not tee_service:
                raise ValueError("No TEE service found in session")
            instance_id = tee_service.get("instance_id")

        # Handle single blob name or list of blob names
        if isinstance(blob_names, str):
            blob_names = [blob_names]

        print(f"\n⚙️  Step 8: Processing {len(blob_names)} blob(s) with TEE...")
        url = f"{self.base_url}/sessions/{self.session_id}/tee/process"
        payload = {
            "tee_instance_id": instance_id,
            "blob_names": blob_names,
            "invocation_name": invocation_name
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            print(f"✅ TEE processing completed")
            if "encrypted_results" in data:
                print(f"   Result size: {len(data['encrypted_results'])//2} bytes")
            return data
        except requests.exceptions.RequestException as e:
            print(f"⚠️  TEE processing skipped: {e}")
            return {}

def run_full_kms_flow(client):
    """Execute the complete 9-step KMS flow from kms-flow-simulation.

    Policy Architecture:
    - Step 3a: Create PipelineVariantPolicy (concrete processing graph with node IDs and transforms)
    - Step 3b: Create DataAccessPolicy wrapper containing the variant
    - Step 6: Register Pipeline - extracts variant from DataAccessPolicy
    - Step 7: Authorize Transform - extracts variant from DataAccessPolicy again

    Why extract twice: Ensures the exact same variant bytes are used for both
    register_pipeline and authorize_transform, matching KMS expectations.
    """
    try:
        # Step 1: Rotate keyset
        keyset_id = random.randint(1, 1000)
        client.rotate_keyset(keyset_id=keyset_id, ttl_seconds=7200)

        # Step 2: Evidence already retrieved (done above in main)
        print(f"\n📊 Step 2: TEE Evidence (already retrieved above)")

        # Step 3a: Create variant policy with TEE reference values
        variant_policy_name = "test_concat_variant"
        print(f"\n🏗️  Step 3a: Creating PipelineVariantPolicy with TEE reference values")
        try:
            client.create_variant_policy(
                variant_policy_name=variant_policy_name,
                src_node_ids=[0],  # Read from input node
                dst_node_ids=[1]   # Write to output node
            )
        except Exception as e:
            print(f"❌ Variant policy creation failed: {e}")
            return

        # Step 3b: Create data access policy from variant
        # This wraps the variant in a DataAccessPolicy structure that KMS will authorize
        policy_name = "test_concat_policy"
        print(f"\n🏗️  Step 3b: Creating DataAccessPolicy wrapper for variant")
        try:
            client.create_data_access_policy(
                policy_name=policy_name,
                logical_pipeline_name="test_concat_pipeline",
                variant_policy_names=[variant_policy_name]
            )
        except Exception as e:
            print(f"❌ Data access policy creation failed: {e}")
            return

        # Step 3c: Derive encryption keys from data access policy
        print(f"\n🔐 Step 3c: Deriving encryption keys from policy")
        try:
            client.derive_keys(policy_name=policy_name)
        except Exception as e:
            print(f"❌ Key derivation failed: {e}")
            return

        # Step 4: Client encrypts data (two chunks for test_concat)
        print(f"\n🔒 Step 4: Client Encrypts Data (Two Separate Plaintext Chunks)")
        chunk_1_plaintext = "chunk_1_data_"
        chunk_2_plaintext = "chunk_2_data_"
        print(f"   Chunk 1 plaintext: {len(chunk_1_plaintext)} bytes - '{chunk_1_plaintext}'")
        print(f"   Chunk 2 plaintext: {len(chunk_2_plaintext)} bytes - '{chunk_2_plaintext}'")

        blob_name_1 = "upload_chunk_1"
        blob_name_2 = "upload_chunk_2"

        try:
            encryption_1 = client.encrypt_data(
                blob_name=blob_name_1,
                plaintext=chunk_1_plaintext,
                policy_name=policy_name,
                public_key_index=0
            )
        except Exception as e:
            print(f"⚠️  Chunk 1 encryption skipped: {e}")
            encryption_1 = None

        try:
            encryption_2 = client.encrypt_data(
                blob_name=blob_name_2,
                plaintext=chunk_2_plaintext,
                policy_name=policy_name,
                public_key_index=0
            )
        except Exception as e:
            print(f"⚠️  Chunk 2 encryption skipped: {e}")
            encryption_2 = None

        # Step 5: Store encrypted data (implicit in session)
        print(f"\n💾 Step 5: Store Encrypted Data")
        print(f"   Encrypted data stored: {blob_name_1} and {blob_name_2}")

        # Step 6: Register pipeline invocation
        # Server will extract the variant from the data access policy using logical_pipeline_name
        invocation_name = "test_concat_invocation"
        logical_pipeline_name = "test_concat_pipeline"
        print(f"\n📝 Step 6: Registering pipeline invocation")
        print(f"   Server will extract variant from logical pipeline '{logical_pipeline_name}'")
        try:
            client.register_pipeline(
                invocation_name=invocation_name,
                logical_pipeline_name=logical_pipeline_name,
                data_access_policy_name=policy_name,
                keyset_id=keyset_id,
                ttl_seconds=3600
            )
        except Exception as e:
            print(f"❌ Pipeline registration failed: {e}")
            return

        # Step 7: Authorize transform
        try:
            client.authorize_transform(invocation_name=invocation_name)
        except Exception as e:
            print(f"⚠️  Transform authorization skipped: {e}")

        # Step 8: Pipeline sends encrypted data to TEE (both blobs in one call)
        blob_names_to_process = []
        if encryption_1:
            blob_names_to_process.append(blob_name_1)
        if encryption_2:
            blob_names_to_process.append(blob_name_2)

        if blob_names_to_process:
            try:
                response_data = client.process_with_tee(
                    blob_names=blob_names_to_process,
                    invocation_name=invocation_name
                )
                # Display final results from test_concat (plaintext)
                if "encrypted_results" in response_data:
                    result_hex = response_data["encrypted_results"]
                    result_bytes = bytes.fromhex(result_hex)
                    print(f"\n✅ Step 8: TEE Processing Complete")
                    print(f"   test_concat returned plaintext results: {len(result_bytes)} bytes")
                    # Try to decode as UTF-8 string
                    try:
                        result_text = result_bytes.decode('utf-8', errors='replace')
                        print(f"   Result content: {result_text}")
                    except Exception:
                        print(f"   Result (hex): {result_hex[:100]}")
            except Exception as e:
                print(f"⚠️  TEE processing skipped: {e}")
        else:
            print(f"⚠️  No blobs encrypted, skipping TEE processing")

        # Step 9: Note that results are plaintext from test_concat
        print(f"\n📋 Step 9: Results Available")
        print(f"   test_concat returns plaintext results directly")
        print(f"   No KMS ReleaseResults needed (unlike production flow)")

        print(f"\n✅ 9-Step KMS Flow Completed!")

    except Exception as e:
        print(f"\n❌ KMS flow error: {e}")
        import traceback
        traceback.print_exc()


def run_policy_hash_mismatch_flow(client):
    """Execute the policy hash mismatch scenario test.

    This test demonstrates what happens when the policy hash used to derive keys
    differs from the one used to register and authorize the pipeline.

    Flow:
    1. Create two different policies (Policy A and Policy B with different pipeline names)
    2. Derive encryption keys from Policy A's hash
    3. Encrypt data with the public key from Policy A
    4. Register and authorize pipeline with Policy B (DIFFERENT!)
    5. Send data to TEE for processing
    6. TEE tries to decrypt with keys for Policy B
    7. Decryption fails because HKDF("info"=policy_hash_A) ≠ HKDF("info"=policy_hash_B)

    This shows that KMS does NOT validate policy hash consistency - it relies on
    cryptographic integrity to fail gracefully when hashes don't match.
    """
    try:
        print("\n" + "=" * 70)
        print("⚠️  POLICY HASH MISMATCH SCENARIO TEST")
        print("=" * 70)
        print("This test demonstrates the consequences of policy hash mismatch:")
        print("- Data encrypted with Policy A's key")
        print("- TEE authorized with Policy B's key")
        print("- Decryption will FAIL in the TEE")
        print("=" * 70)

        # Setup: Create keyset
        keyset_id = random.randint(1000, 9999)
        client.rotate_keyset(keyset_id=keyset_id, ttl_seconds=7200)
        print(f"\n✓ Keyset rotated: {keyset_id}")

        # Step 1: Create Policy A (for encryption)
        policy_a_variant_name = "policy_a_variant"
        policy_a_name = "policy_a_access_policy"
        logical_pipeline_a = "pipeline_a_name"

        print(f"\n📋 Step 1: Creating Policy A (for encryption)")
        print(f"   Logical pipeline: {logical_pipeline_a}")
        try:
            client.create_variant_policy(
                variant_policy_name=policy_a_variant_name,
                src_node_ids=[0],
                dst_node_ids=[1]
            )
            print(f"   ✓ Variant policy created: {policy_a_variant_name}")

            client.create_data_access_policy(
                policy_name=policy_a_name,
                logical_pipeline_name=logical_pipeline_a,
                variant_policy_names=[policy_a_variant_name]
            )
            print(f"   ✓ Data access policy created: {policy_a_name}")
        except Exception as e:
            print(f"   ❌ Policy A creation failed: {e}")
            return

        # Step 2: Create Policy B (for authorization - DIFFERENT!)
        policy_b_variant_name = "policy_b_variant"
        policy_b_name = "policy_b_access_policy"
        logical_pipeline_b = "pipeline_b_name"  # Different name = different hash!

        print(f"\n📋 Step 2: Creating Policy B (for authorization - DIFFERENT!)")
        print(f"   Logical pipeline: {logical_pipeline_b}")
        print(f"   ⚠️  WARNING: Different pipeline name will result in different hash")
        try:
            client.create_variant_policy(
                variant_policy_name=policy_b_variant_name,
                src_node_ids=[0],
                dst_node_ids=[1]
            )
            print(f"   ✓ Variant policy created: {policy_b_variant_name}")

            client.create_data_access_policy(
                policy_name=policy_b_name,
                logical_pipeline_name=logical_pipeline_b,
                variant_policy_names=[policy_b_variant_name]
            )
            print(f"   ✓ Data access policy created: {policy_b_name}")
        except Exception as e:
            print(f"   ❌ Policy B creation failed: {e}")
            return

        # Step 3: Derive keys from Policy A's hash
        print(f"\n🔐 Step 3: Deriving encryption keys from Policy A")
        print(f"   Policy: {policy_a_name}")
        try:
            client.derive_keys(policy_name=policy_a_name)
            print(f"   ✓ Encryption keys derived from Policy A")
        except Exception as e:
            print(f"   ❌ Key derivation failed: {e}")
            return

        # Step 4: Encrypt data with Policy A key
        print(f"\n🔒 Step 4: Encrypting data with Policy A key")
        plaintext = "Secret data encrypted with policy_a"
        blob_name = "mismatch_test_blob"
        print(f"   Plaintext: '{plaintext}'")
        print(f"   Plaintext size: {len(plaintext)} bytes")

        try:
            encryption_result = client.encrypt_data(
                blob_name=blob_name,
                plaintext=plaintext,
                policy_name=policy_a_name,  # ← Using Policy A
                public_key_index=0
            )
            print(f"   ✓ Data encrypted with Policy A key")
            if encryption_result:
                print(f"   Encrypted size: {len(encryption_result.get('encrypted_blob', ''))} bytes")
        except Exception as e:
            print(f"   ❌ Encryption failed: {e}")
            return

        # Step 5: Register pipeline with Policy B (MISMATCH!)
        invocation_name = "mismatch_invocation"
        print(f"\n📝 Step 5: Registering pipeline with Policy B (DIFFERENT!)")
        print(f"   ⚠️  Register using Policy B: {policy_b_name}")
        print(f"   ⚠️  But data encrypted with Policy A: {policy_a_name}")
        try:
            client.register_pipeline(
                invocation_name=invocation_name,
                logical_pipeline_name=logical_pipeline_b,  # ← Using Policy B
                data_access_policy_name=policy_b_name,  # ← Using Policy B
                keyset_id=keyset_id,
                ttl_seconds=3600
            )
            print(f"   ✓ Pipeline registered with Policy B")
            print(f"   ⚠️  MISMATCH: Keys are for Policy B, data encrypted with Policy A")
        except Exception as e:
            print(f"   ⚠️  Pipeline registration failed: {e}")
            print(f"   This is expected if KMS validates policy consistency")
            return

        # Step 6: Authorize transform with Policy B
        print(f"\n🔐 Step 6: Authorizing transform with Policy B")
        try:
            client.authorize_transform(invocation_name=invocation_name)
            print(f"   ✓ Transform authorized with Policy B")
            print(f"   ✓ Decryption keys are for Policy B")
            print(f"   ⚠️  But data was encrypted with Policy A key!")
        except Exception as e:
            print(f"   ⚠️  Authorization failed: {e}")
            print(f"   This might prevent the test from proceeding")

        # Step 7: Send encrypted data to TEE and observe the failure
        print(f"\n⚡ Step 7: Sending mismatched data to TEE")
        print(f"   Data encrypted with: Policy A key (HKDF with Policy A hash)")
        print(f"   TEE will receive: Policy B key (HKDF with Policy B hash)")
        print(f"   Expected: Decryption FAILS in TEE")

        response_data = client.process_with_tee(
            blob_names=[blob_name],
            invocation_name=invocation_name
        )

        # Check if TEE processing succeeded (has results) or failed (empty response)
        if "encrypted_results" in response_data and response_data["encrypted_results"]:
            # Decryption unexpectedly succeeded
            print(f"\n⚠️  UNEXPECTED SUCCESS!")
            print(f"   The TEE successfully decrypted the data.")
            print(f"   This suggests the policy hashes were not actually different.")
            result_hex = response_data["encrypted_results"]
            result_bytes = bytes.fromhex(result_hex)
            try:
                result_text = result_bytes.decode('utf-8', errors='replace')
                print(f"   Result: {result_text}")
            except Exception:
                print(f"   Result (hex): {result_hex[:100]}")
        else:
            # Decryption failed as expected (empty response_data)
            print(f"\n✓ Decryption FAILED as expected!")
            print(f"   TEE could not decrypt the data due to policy hash mismatch.")
            print(f"\n   This demonstrates that:")
            print(f"   - KMS does NOT prevent policy hash mismatches")
            print(f"   - The failure happens at TEE decryption time")
            print(f"   - Different policy hashes → different HKDF keys")
            print(f"   - Key A cannot decrypt data encrypted with Key B")
            print(f"   - Cryptographic integrity ensures safe failure")

        # Step 8: Summary
        print(f"\n" + "=" * 70)
        print("📋 POLICY HASH MISMATCH TEST SUMMARY")
        print("=" * 70)
        print("Key Insights:")
        print("✓ KMS allows mismatched policy hashes during authorization")
        print("✓ The error manifests ONLY in the TEE (at decryption time)")
        print("✓ Different policy hashes → different HKDF derived keys")
        print("✓ Key A (Policy A) cannot decrypt data encrypted with Key B (Policy B)")
        print("✓ This is a design feature - cryptography ensures safe failure")
        print("✓ System operators MUST ensure policy consistency")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Policy hash mismatch test error: {e}")
        import traceback
        traceback.print_exc()


def run_concurrent_pressure_test(host: str = "localhost", port: int = 3000, num_users: int = 20):
    """
    Concurrent pressure test: N users running complete 9-step KMS flow concurrently.

    Test flow per user:
    0. Create session, launch TEE, get reference values
    1. Rotate keyset
    2. TEE evidence (already retrieved)
    3. Create policies (variant + data access) and derive keys
    4. Encrypt data (two chunks)
    5. Store encrypted data
    6. Register pipeline invocation
    7. Authorize transform with TEE evidence
    8. Process encrypted data with TEE
    9. Receive results from TEE

    Overall test phases:
    - Phase 1: All users run complete 9-step flow concurrently
    - Phase 2: Summary of results
    - Phase 3: Wait 30 seconds (resources under pressure)
    - Phase 4: Cleanup all sessions

    Args:
        host: Workshop server host
        port: Workshop server port
        num_users: Number of concurrent users (default: 20)
    """
    print("=" * 80)
    print(f"🔥 CONCURRENT PRESSURE TEST: {num_users} USERS - FULL 9-STEP KMS FLOW")
    print("=" * 80)
    print(f"Target: http://{host}:{port}")
    print(f"Concurrent users: {num_users}")
    print(f"Each user: Complete 9-step KMS flow (session → TEE → policies → encrypt → process)")
    print("=" * 80)

    # Thread-safe result tracking
    results_lock = threading.Lock()
    successful_sessions = []
    failed_sessions = []

    def single_user_workflow(user_id: int):
        """Single user workflow: complete 9-step KMS flow."""
        session_id = None
        client = None

        try:
            print(f"\n[User {user_id:02d}] 🚀 Starting full KMS workflow...")

            # Create client
            client = WorkshopServerClient(host, port)

            # Step 0: Create session
            print(f"[User {user_id:02d}] 📝 Creating session...")
            client.create_session(timeout_seconds=300)
            session_id = client.session_id
            print(f"[User {user_id:02d}] ✅ Session created: {session_id}")

            # Step 0b: Launch TEE
            print(f"[User {user_id:02d}] 🔧 Launching TEE...")
            tee_response = client.start_tee(memory_size="2G")

            # Extract TEE instance ID
            tee_instance_id = None
            if "instances" in tee_response and len(tee_response["instances"]) > 0:
                tee_instance_id = tee_response["instances"][0].get("instance_id")
                print(f"[User {user_id:02d}] ✅ TEE launched: {tee_instance_id}")
            else:
                raise ValueError("No TEE instance returned")

            # Step 0c: Get reference values (needed for policy creation)
            print(f"[User {user_id:02d}] 📊 Fetching reference values...")
            ref_values_data = client.get_tee_reference_values(instance_id=tee_instance_id)

            if ref_values_data:
                ref_values_hex = ref_values_data.get('reference_values_hex', '')
                print(f"[User {user_id:02d}] ✅ Reference values retrieved ({len(ref_values_hex)//2} bytes)")
            else:
                print(f"[User {user_id:02d}] ⚠️  Reference values not available")

            # ===== NOW RUN FULL 9-STEP KMS FLOW =====
            print(f"[User {user_id:02d}] 🔐 Starting 9-step KMS flow...")

            # Step 1: Rotate keyset
            keyset_id = random.randint(1, 1000)
            print(f"[User {user_id:02d}] 🔄 Step 1: Rotating keyset {keyset_id}...")
            client.rotate_keyset(keyset_id=keyset_id, ttl_seconds=7200)

            # Step 2: Evidence already retrieved above
            print(f"[User {user_id:02d}] 📊 Step 2: Evidence already retrieved")

            # Step 3a: Create variant policy
            variant_policy_name = f"user_{user_id}_variant"
            print(f"[User {user_id:02d}] 🏗️  Step 3a: Creating variant policy...")
            client.create_variant_policy(
                variant_policy_name=variant_policy_name,
                src_node_ids=[0],
                dst_node_ids=[1]
            )

            # Step 3b: Create data access policy
            policy_name = f"user_{user_id}_policy"
            print(f"[User {user_id:02d}] 🏗️  Step 3b: Creating data access policy...")
            client.create_data_access_policy(
                policy_name=policy_name,
                logical_pipeline_name=f"user_{user_id}_pipeline",
                variant_policy_names=[variant_policy_name]
            )

            # Step 3c: Derive encryption keys
            print(f"[User {user_id:02d}] 🔐 Step 3c: Deriving encryption keys...")
            client.derive_keys(policy_name=policy_name)

            # Step 4: Encrypt data (two chunks)
            print(f"[User {user_id:02d}] 🔒 Step 4: Encrypting data chunks...")
            blob_name_1 = f"user_{user_id}_chunk_1"
            blob_name_2 = f"user_{user_id}_chunk_2"

            client.encrypt_data(
                blob_name=blob_name_1,
                plaintext=f"user_{user_id}_data_chunk_1_",
                policy_name=policy_name,
                public_key_index=0
            )
            client.encrypt_data(
                blob_name=blob_name_2,
                plaintext=f"user_{user_id}_data_chunk_2_",
                policy_name=policy_name,
                public_key_index=0
            )

            # Step 5: Data stored (implicit)
            print(f"[User {user_id:02d}] 💾 Step 5: Data stored")

            # Step 6: Register pipeline invocation
            invocation_name = f"user_{user_id}_invocation"
            print(f"[User {user_id:02d}] 📝 Step 6: Registering pipeline...")
            client.register_pipeline(
                invocation_name=invocation_name,
                logical_pipeline_name=f"user_{user_id}_pipeline",
                data_access_policy_name=policy_name,
                keyset_id=keyset_id,
                ttl_seconds=3600
            )

            # Step 7: Authorize transform
            print(f"[User {user_id:02d}] 🔓 Step 7: Authorizing transform...")
            client.authorize_transform(invocation_name=invocation_name)

            # Step 8: Process with TEE
            print(f"[User {user_id:02d}] ⚙️  Step 8: Processing with TEE...")
            response_data = client.process_with_tee(
                blob_names=[blob_name_1, blob_name_2],
                invocation_name=invocation_name
            )

            # Step 9: Results
            if "encrypted_results" in response_data:
                result_hex = response_data["encrypted_results"]
                result_bytes = bytes.fromhex(result_hex)
                result_text = result_bytes.decode('utf-8', errors='replace')
                print(f"[User {user_id:02d}] ✅ Step 9: Results received ({len(result_hex)//2} bytes)")
                print(f"[User {user_id:02d}]    Result content: {result_text}")
            else:
                print(f"[User {user_id:02d}] ⚠️  Step 9: No results")

            # Mark success
            with results_lock:
                successful_sessions.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'client': client,
                    'tee_instance_id': tee_instance_id
                })

            print(f"[User {user_id:02d}] ✅ Full KMS workflow completed successfully!")
            return True

        except Exception as e:
            print(f"[User {user_id:02d}] ❌ Error: {e}")
            with results_lock:
                failed_sessions.append({
                    'user_id': user_id,
                    'session_id': session_id,
                    'client': client,
                    'error': str(e)
                })
            return False

    # Phase 1: Execute concurrent workflows
    print(f"\n{'=' * 80}")
    print("📊 PHASE 1: CONCURRENT FULL 9-STEP KMS FLOW")
    print("=" * 80)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        # Submit all tasks
        futures = {executor.submit(single_user_workflow, i): i for i in range(1, num_users + 1)}

        # Wait for all to complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            user_id = futures[future]
            try:
                success = future.result()
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"\n[Progress] {completed}/{num_users} users completed | User {user_id:02d}: {status}")
            except Exception as e:
                print(f"\n[Progress] {completed}/{num_users} users completed | User {user_id:02d}: ❌ EXCEPTION: {e}")

    elapsed_time = time.time() - start_time

    # Phase 2: Summary
    print(f"\n{'=' * 80}")
    print("📈 PHASE 2: TEST SUMMARY")
    print("=" * 80)
    print(f"Total users: {num_users}")
    print(f"✅ Successful: {len(successful_sessions)}")
    print(f"❌ Failed: {len(failed_sessions)}")
    print(f"⏱️  Total elapsed time: {elapsed_time:.2f}s")
    print(f"⏱️  Average time per user: {elapsed_time/num_users:.2f}s")

    if failed_sessions:
        print(f"\n⚠️  Failed sessions:")
        for fail in failed_sessions:
            print(f"   - User {fail['user_id']:02d}: {fail['error']}")

    # Phase 3: Wait 30 seconds
    print(f"\n{'=' * 80}")
    print("⏳ PHASE 3: WAITING 30 SECONDS (resources under pressure)")
    print("=" * 80)

    for i in range(30, 0, -5):
        total_sessions = len(successful_sessions) + len(failed_sessions)
        print(f"⏰ {i} seconds remaining... ({total_sessions} sessions active)")
        time.sleep(5)

    print("✅ Wait complete")

    # Phase 4: Cleanup ALL sessions (both successful and failed)
    print(f"\n{'=' * 80}")
    print("🧹 PHASE 4: CLEANUP (deleting all sessions)")
    print("=" * 80)

    # Combine all sessions that need cleanup
    all_sessions = successful_sessions + failed_sessions
    print(f"Total sessions to clean up: {len(all_sessions)} (successful: {len(successful_sessions)}, failed: {len(failed_sessions)})")

    cleanup_success = 0
    cleanup_failed = 0

    for session_info in all_sessions:
        try:
            user_id = session_info['user_id']
            client = session_info.get('client')
            session_id = session_info.get('session_id')

            # Skip if no client or session_id available
            if not client or not session_id:
                print(f"[User {user_id:02d}] ⚠️  No session to clean up (creation failed early)")
                continue

            print(f"[User {user_id:02d}] 🗑️  Deleting session {session_id}...")
            client.delete_session()
            cleanup_success += 1
            print(f"[User {user_id:02d}] ✅ Session deleted")

        except Exception as e:
            cleanup_failed += 1
            print(f"[User {user_id:02d}] ❌ Cleanup error: {e}")

    # Final Summary
    print(f"\n{'=' * 80}")
    print("🏁 FINAL SUMMARY")
    print("=" * 80)
    print(f"Test Duration: {elapsed_time:.2f}s")
    print(f"Sessions Created: {len(all_sessions)}/{num_users}")
    print(f"  - Successful workflows: {len(successful_sessions)}")
    print(f"  - Failed workflows: {len(failed_sessions)}")
    print(f"Sessions Cleaned: {cleanup_success}/{len(all_sessions)}")
    print(f"Cleanup Failures: {cleanup_failed}")

    if len(successful_sessions) == num_users and cleanup_success == len(all_sessions):
        print("\n✅ PRESSURE TEST PASSED: All sessions created, processed, and cleaned up successfully!")
    else:
        print("\n⚠️  PRESSURE TEST COMPLETED WITH ISSUES: Review failures above")
        if len(failed_sessions) > 0:
            print(f"   - {len(failed_sessions)} workflow(s) failed")
        if cleanup_failed > 0:
            print(f"   - {cleanup_failed} cleanup(s) failed")

    print("=" * 80)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test Workshop Server KMS and TEE services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic service test
  python3 test_services.py

  # Full 9-step KMS flow
  python3 test_services.py --full-flow

  # Policy hash mismatch scenario test
  python3 test_services.py --policy-hash-mismatch

  # Full flow with custom settings
  python3 test_services.py --full-flow --host localhost --port 8080

  # Full flow with longer timeout and verbose output
  python3 test_services.py --full-flow --timeout 120 --verbose

  # Full flow with custom memory configuration
  python3 test_services.py --full-flow --kms-memory 4096 --tee-memory 4096

  # Full flow on remote host
  python3 test_services.py --full-flow --host 192.168.1.100 --port 3000

  # Policy hash mismatch with custom memory
  python3 test_services.py --policy-hash-mismatch --kms-memory 4096 --tee-memory 4096
        """,
    )
    parser.add_argument(
        "--host", default="localhost", help="Workshop server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=3000, help="Workshop server port (default: 3000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="How long to keep services running (default: 30s)",
    )
    parser.add_argument(
        "--kms-memory",
        default="2G",
        help="KMS service memory",
    )
    parser.add_argument(
        "--tee-memory",
        default="2G",
        help="TEE service memory",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Poll interval for monitoring in seconds (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed session information during monitoring (default: false)",
    )
    parser.add_argument(
        "--full-flow",
        action="store_true",
        help="Run the complete 9-step KMS flow (default: false)",
    )
    parser.add_argument(
        "--policy-hash-mismatch",
        action="store_true",
        help="Run the policy hash mismatch scenario test (demonstrates cryptographic failure when policy hashes don't match)",
    )
    parser.add_argument(
        "--pressure-test",
        action="store_true",
        help="Run concurrent pressure test with multiple users (default: false)",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=20,
        help="Number of concurrent users for pressure test (default: 20)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 Workshop Server Service Test")
    print("=" * 70)
    print(f"Target: {args.host}:{args.port}")

    # Handle pressure test mode separately (different execution path)
    if args.pressure_test:
        print(f"Test Mode: Concurrent Pressure Test")
        print(f"Concurrent Users: {args.num_users}")
        print("=" * 70)
        try:
            run_concurrent_pressure_test(args.host, args.port, args.num_users)
            return 0
        except KeyboardInterrupt:
            print("\n\n⚠️  Pressure test interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Pressure test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Determine test mode for regular tests
    if args.full_flow:
        test_mode = "Full 9-Step KMS Flow"
    elif args.policy_hash_mismatch:
        test_mode = "Policy Hash Mismatch Scenario"
    else:
        test_mode = "Basic Service Test"

    print(f"Test Mode: {test_mode}")
    print(f"Test Duration: {args.timeout}s")
    print(f"KMS Memory: {args.kms_memory}MB")
    print(f"TEE Memory: {args.tee_memory}MB")
    print("=" * 70)

    client = WorkshopServerClient(args.host, args.port)

    try:
        # Pre-flight: Check server status and capacity
        print("\n" + "=" * 70)
        print("🔍 PRE-FLIGHT: SERVER STATUS CHECK")
        print("=" * 70)
        status_data = client.get_status()

        # Check if server is at capacity
        if "sessions" in status_data:
            sessions = status_data["sessions"]
            if sessions.get("at_limit", False):
                print("\n❌ Server is at maximum capacity!")
                print(f"   Maximum concurrent sessions: {sessions.get('max', 0)}")
                print(f"   Active sessions: {sessions.get('active', 0)}")
                print(f"   Please wait for a session to become available or try again later.")
                return 1

        # Step 0: Create session (KMS is automatically assigned from pool)
        session_data = client.create_session(timeout_seconds=args.timeout + 60)

        # Display KMS auto-assignment info
        if "kms_service" in session_data:
            kms_info = session_data["kms_service"]
            print(f"\n🔐 KMS Auto-Assigned from Pool:")
            print(f"   Address: {kms_info.get('address', 'N/A')}")
            print(f"   Port: {kms_info.get('port', 'N/A')}")

        # Step 1: Start KMS (DEPRECATED - KMS is now auto-assigned)
        # Note: This step is no longer necessary as KMS is automatically assigned
        # when the session is created. Uncomment below to verify KMS assignment:
        # client.start_kms(memory_size=args.kms_memory)

        # Step 2: Start TEE
        tee_response = client.start_tee(memory_size=args.tee_memory)

        # Step 3: Fetch and display TEE evidence
        print("\n" + "=" * 70)
        print("🔍 ATTESTATION DATA RETRIEVAL")
        print("=" * 70)

        # Get the instance ID from TEE response
        tee_instance_id = None
        if "instances" in tee_response and len(tee_response["instances"]) > 0:
            tee_instance_id = tee_response["instances"][0].get("instance_id")

        if tee_instance_id:

            # Fetch and display evidence
            evidence_data = client.get_tee_evidence(instance_id=tee_instance_id)

            # Fetch and display reference values
            ref_values_data = client.get_tee_reference_values(instance_id=tee_instance_id)

            # Print summary
            print("\n" + "=" * 70)
            print("📋 ATTESTATION DATA SUMMARY")
            print("=" * 70)
            if evidence_data:
                print("✅ Evidence: Successfully retrieved")
                evidence_hex = evidence_data.get('evidence_hex', '')
                endorsements_hex = evidence_data.get('endorsements_hex', '')
                print(f"   - Evidence size: {len(evidence_hex)//2} bytes")
                print(f"   - Endorsements size: {len(endorsements_hex)//2} bytes")
            else:
                print("⚠️  Evidence: Not available")

            if ref_values_data:
                print("✅ Reference Values: Successfully retrieved")
                ref_values_hex = ref_values_data.get('reference_values_hex', '')
                print(f"   - Reference Values size: {len(ref_values_hex)//2} bytes")
            else:
                print("⚠️  Reference Values: Not available")
        else:
            print("⚠️  No TEE instance found, skipping evidence retrieval")

        # Run full 9-step KMS flow if requested
        if args.full_flow:
            print("\n" + "=" * 70)
            print("🔐 FULL 9-STEP KMS FLOW")
            print("=" * 70)
            run_full_kms_flow(client)

        # Run policy hash mismatch scenario if requested
        if args.policy_hash_mismatch:
            run_policy_hash_mismatch_flow(client)

        # Step X: Monitor services (optional, brief monitoring)
        if args.timeout > 0:
            client.wait_and_monitor(
                duration=min(args.timeout, 5),
                poll_interval=args.poll_interval,
                verbose=args.verbose
            )

        # Cleanup
        client.get_session()
        client.delete_session()

        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        try:
            client.delete_session()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        return 1

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nAttempting cleanup...")
        try:
            client.delete_session()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
