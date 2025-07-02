import os
import time
from typing import Dict, List, Optional

import requests
import runpod
from fastapi import HTTPException


class RunPodManager:
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY environment variable is not set")
        runpod.api_key = self.api_key

        # Load configuration from environment variables with defaults
        self.config = {
            "min_vram_gb": int(os.getenv("RUNPOD_MIN_VRAM_GB", "16")),
            "image": os.getenv("RUNPOD_IMAGE", "koboldai/koboldcpp:latest"),
            "model_url": os.getenv("RUNPOD_DEFAULT_MODEL"),
            "hf_token": os.getenv("HF_TOKEN"),
            "port": os.getenv("RUNPOD_PORT", "5001"),
            "cloud_type": os.getenv("RUNPOD_CLOUD_TYPE", "SECURE").upper(),
            "context_size": os.getenv("KCPP_CONTEXT_SIZE", "4096"),
            "gpu_layers": os.getenv("KCPP_GPU_LAYERS", "9999"),
            "multi_user": os.getenv("KCPP_MULTI_USER", "20"),
        }

    def find_gpu(self) -> Optional[Dict]:
        """Find the cheapest suitable GPU."""
        try:
            gpus = runpod.get_gpus()
            suitable_gpus = []

            for gpu in gpus:
                if gpu["memoryInGb"] < self.config["min_vram_gb"]:
                    continue

                gpu_details = runpod.get_gpu(gpu["id"])
                if not gpu_details or gpu_details["manufacturer"] != "Nvidia":
                    continue

                price = (
                    gpu_details.get("securePrice", float("inf"))
                    if self.config["cloud_type"] == "SECURE"
                    else gpu_details.get("communityPrice", float("inf"))
                )

                if price <= 0:
                    continue

                suitable_gpus.append(
                    {
                        "id": gpu["id"],
                        "displayName": gpu["displayName"],
                        "memoryInGb": gpu["memoryInGb"],
                        "price": price,
                    }
                )

            if not suitable_gpus:
                return None

            return min(suitable_gpus, key=lambda x: x["price"])

        except Exception as e:
            print(f"Error finding GPU: {str(e)}")
            return None

    def create_pod(self) -> Optional[Dict]:
        """Create a new RunPod instance."""
        try:
            # Terminate existing pods
            for pod in self.list_pods():
                self.terminate_pod(pod["pod_id"])
                time.sleep(5)

            gpu = self.find_gpu()
            if not gpu:
                raise Exception("No suitable GPUs available")

            pod = runpod.create_pod(
                name="language-model-pod",
                image_name=self.config["image"],
                gpu_type_id=gpu["id"],
                cloud_type=self.config["cloud_type"],
                ports=f"{self.config['port']}/http",
                env={
                    "KCPP_MODEL": self.config["model_url"],
                    "HF_TOKEN": self.config.get("hf_token", os.getenv("HF_TOKEN")),
                    "KCPP_ARGS": (
                        f"--contextsize {self.config['context_size']} "
                        "--chatcompletionsadapter AutoGuess "
                        "--ttsgpu --multiplayer --sdquant "
                        "--usecublas mmq "
                        f"--gpulayers {self.config['gpu_layers']} "
                        f"--multiuser {self.config['multi_user']} "
                        "--flashattention --ignoremissing"
                    ),
                },
            )

            if not pod or "id" not in pod:
                raise Exception("Invalid response from RunPod API")

            pod_id = pod["id"]
            endpoint_base_url = f"https://{pod_id}-5001.proxy.runpod.net"
            
            # Set the environment variable immediately without /v1
            os.environ["RUNPOD_ENDPOINT_URL"] = f"{endpoint_base_url}/v1"
            
            return {
                "pod_id": pod_id,
                "endpoint_url": f"{endpoint_base_url}/v1",
                "endpoint_base_url": endpoint_base_url,
                "status": "STARTING",
                "status_message": "Pod is starting...",
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate the RunPod instance."""
        try:
            # Verify pod exists before attempting to terminate
            pod_status = runpod.get_pod(pod_id)
            if not pod_status:
                raise Exception(f"Pod {pod_id} not found")

            result = runpod.terminate_pod(pod_id)
            if not result:
                raise Exception("Pod termination failed")
            return True

        except Exception as e:
            print(f"Error terminating pod: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to terminate pod: {str(e)}"
            )

    def list_pods(self) -> List[Dict]:
        """List all RunPod instances."""
        try:
            pods = runpod.get_pods()
            if not pods:
                return []

            return [
                {
                    "pod_id": pod["id"],
                    "name": pod.get("name", "Unknown"),
                    "status": pod.get("desiredStatus", "Unknown"),
                    "endpoint_url": f"https://{pod['id']}-5001.proxy.runpod.net/v1",
                }
                for pod in pods
            ]
        except Exception as e:
            print(f"Error listing pods: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to list pods: {str(e)}"
            )

    def check_endpoint_status(self, pod_id: str) -> Dict:
        """Check if the LLM endpoint is ready."""
        url = f"https://{pod_id}-5001.proxy.runpod.net/v1"
        base_url = f"https://{pod_id}-5001.proxy.runpod.net"

        try:
            # First check if the pod exists and get its status
            pod_status = runpod.get_pod(pod_id)
            if not pod_status:
                return {
                    "is_ready": False,
                    "status": "NOT_FOUND",
                    "status_message": "Pod not found"
                }
            
            pod_desired_status = pod_status.get("desiredStatus", "Unknown")
            pod_runtime_status = pod_status.get("runtime", {}).get("status", "Unknown")
            
            # Check if base endpoint is accessible (without /v1)
            try:
                base_response = requests.get(base_url, timeout=10)
                base_accessible = base_response.status_code == 200
            except:
                base_accessible = False
            
            # Check if LLM endpoint is ready
            try:
                response = requests.get(f"{url}/models", timeout=30)
                llm_ready = response.status_code == 200 and response.text.strip()
            except:
                llm_ready = False

            if llm_ready:
                # LLM is fully ready
                os.environ["RUNPOD_ENDPOINT_URL"] = url
                return {
                    "is_ready": True,
                    "status": "RUNNING",
                    "status_message": "Pod is running and LLM is ready",
                    "pod_status": pod_desired_status,
                    "runtime_status": pod_runtime_status
                }
            elif base_accessible:
                # Pod is running but LLM not ready yet
                return {
                    "is_ready": False,
                    "status": "INITIALIZING", 
                    "status_message": "Pod is running, LLM is initializing",
                    "pod_status": pod_desired_status,
                    "runtime_status": pod_runtime_status
                }
            elif pod_desired_status == "RUNNING" or pod_runtime_status == "RUNNING":
                # Pod should be running but not accessible yet
                return {
                    "is_ready": False,
                    "status": "STARTING",
                    "status_message": "Pod is starting up",
                    "pod_status": pod_desired_status,
                    "runtime_status": pod_runtime_status
                }
            else:
                # Pod is in some other state
                return {
                    "is_ready": False,
                    "status": pod_desired_status,
                    "status_message": f"Pod status: {pod_desired_status}",
                    "pod_status": pod_desired_status,
                    "runtime_status": pod_runtime_status
                }

        except Exception as e:
            print(f"Failed to check endpoint status: {str(e)}")
            # return {
            #     "is_ready": False,
            #     "status": "ERROR",
            #     "status_message": f"Error checking status: {str(e)}"
            # }
