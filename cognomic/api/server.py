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


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
