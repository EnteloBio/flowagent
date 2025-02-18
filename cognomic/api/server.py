"""FastAPI server implementation for Cognomic."""
from typing import Any, Dict, List
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config.settings import settings
from ..core.orchestrator import WorkflowOrchestrator
from ..monitoring.metrics import metrics
from ..security.auth import Token, security_service
from ..utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Cognomic API for bioinformatics workflow automation",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create orchestrator instance
orchestrator = WorkflowOrchestrator()


class WorkflowRequest(BaseModel):
    """Model for workflow creation request."""
    
    name: str
    workflow_type: str
    parameters: Dict[str, Any]


class WorkflowResponse(BaseModel):
    """Model for workflow response."""
    
    id: UUID
    name: str
    status: str
    steps: List[Dict[str, Any]]


@app.get("/")
async def read_root() -> Dict[str, str]:
    return {"message": "Welcome to the Cognomic API!"}


@app.post("/api/v1/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowRequest,
    current_user: Token = Security(security_service.get_current_user)
) -> Dict[str, Any]:
    """Create a new workflow."""
    with metrics.measure_api_latency("/api/v1/workflows"):
        try:
            workflow = await orchestrator.create_workflow(
                name=request.name,
                workflow_type=request.workflow_type,
                parameters=request.parameters,
            )
            metrics.record_api_request(
                "/api/v1/workflows", "POST", 200
            )
            return workflow
        except Exception as e:
            metrics.record_api_request(
                "/api/v1/workflows", "POST", 500
            )
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


@app.get("/api/v1/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: UUID,
    current_user: Token = Security(security_service.get_current_user)
) -> Dict[str, Any]:
    """Get workflow status."""
    with metrics.measure_api_latency(f"/api/v1/workflows/{workflow_id}"):
        try:
            workflow = orchestrator.get_workflow_status(workflow_id)
            if not workflow:
                metrics.record_api_request(
                    f"/api/v1/workflows/{workflow_id}", "GET", 404
                )
                raise HTTPException(
                    status_code=404,
                    detail="Workflow not found"
                )
            
            metrics.record_api_request(
                f"/api/v1/workflows/{workflow_id}", "GET", 200
            )
            return workflow
        except Exception as e:
            metrics.record_api_request(
                f"/api/v1/workflows/{workflow_id}", "GET", 500
            )
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: UUID,
    current_user: Token = Security(security_service.get_current_user)
) -> Dict[str, Any]:
    """Execute a workflow."""
    with metrics.measure_api_latency(f"/api/v1/workflows/{workflow_id}/execute"):
        try:
            workflow = await orchestrator.execute_workflow(workflow_id)
            metrics.record_api_request(
                f"/api/v1/workflows/{workflow_id}/execute", "POST", 200
            )
            return workflow
        except Exception as e:
            metrics.record_api_request(
                f"/api/v1/workflows/{workflow_id}/execute", "POST", 500
            )
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


@app.delete("/api/v1/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: UUID,
    current_user: Token = Security(security_service.get_current_user)
) -> Dict[str, str]:
    """Delete a workflow."""
    with metrics.measure_api_latency(f"/api/v1/workflows/{workflow_id}"):
        try:
            await orchestrator.delete_workflow(workflow_id)
            metrics.record_api_request(
                f"/api/v1/workflows/{workflow_id}", "DELETE", 200
            )
            return {"status": "success"}
        except Exception as e:
            metrics.record_api_request(
                f"/api/v1/workflows/{workflow_id}", "DELETE", 500
            )
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


@app.post("/api/v1/tasks/fastqc")
async def run_fastqc() -> Dict[str, str]:
    # Logic to execute the fastqc task
    # Placeholder for task execution
    return {"status": "fastqc task executed"}


@app.post("/api/v1/tasks/multiqc")
async def run_multiqc() -> Dict[str, str]:
    # Logic to execute the multiqc task
    # Placeholder for task execution
    return {"status": "multiqc task executed"}


@app.post("/api/v1/tasks/kallisto_index")
async def run_kallisto_index() -> Dict[str, str]:
    # Logic to execute the kallisto_index task
    # Placeholder for task execution
    return {"status": "kallisto_index task executed"}


@app.post("/api/v1/tasks/kal_quant")
async def run_kal_quant() -> Dict[str, str]:
    # Logic to execute the kal_quant task
    # Placeholder for task execution
    return {"status": "kal_quant task executed"}


@app.post("/api/v1/tasks/kallisto_multiqc")
async def run_kallisto_multiqc() -> Dict[str, str]:
    # Logic to execute the kallisto_multiqc task
    # Placeholder for task execution
    return {"status": "kallisto_multiqc task executed"}


@app.post("/api/v1/workflows/pseudobulk")
async def run_pseudobulk_workflow() -> Dict[str, str]:
    # Logic to execute the pseudobulk workflow
    # Placeholder for workflow execution
    return {"status": "pseudobulk workflow executed"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
