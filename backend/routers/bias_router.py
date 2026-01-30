"""
Bias Detection Router - API endpoints for Multi-Model Bias Detection & Benchmarking
Owner: Iqra Javed

Feature 6: Multi-Model Bias Detection & Bias Benchmarking Engine

This router provides endpoints to:
1. Configure and run bias detection experiments
2. Analyze bias across demographic groups
3. Generate comprehensive benchmark reports
4. Export results for compliance documentation
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature6_bias_detector import (
    BiasDetectionEngine,
    get_bias_engine,
    MODEL_CONFIGS,
    JOB_ROLES,
    QUALITY_LEVELS,
    DEMOGRAPHIC_NAMES,
    generate_resume
)

router = APIRouter(prefix="/api/bias", tags=["Bias Detection"])

# Store experiment state (in production, use Redis or database)
experiment_state = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0.0,
    "message": "",
    "started_at": None,
    "completed_at": None,
    "results": None
}


# ======================= Pydantic Models =======================

class ExperimentConfig(BaseModel):
    """Configuration for bias detection experiment"""
    models: List[str] = Field(
        default=["gemma-2b", "qwen-1.5b", "tinyllama"],
        description="List of model keys to evaluate"
    )
    job_roles: List[str] = Field(
        default=["Data Analyst", "Software Engineer", "HR Manager"],
        description="List of job roles to test"
    )
    quality_levels: List[str] = Field(
        default=["high", "medium", "low"],
        description="Resume quality levels to include"
    )
    names_per_demographic: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of names per demographic group (1-4)"
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for model access"
    )


class QuickTestConfig(BaseModel):
    """Configuration for quick bias test (single model, minimal data)"""
    model: str = Field(
        default="tinyllama",
        description="Model to test (tinyllama is fastest)"
    )
    job_role: str = Field(
        default="Data Analyst",
        description="Single job role to test"
    )
    quality: str = Field(
        default="high",
        description="Resume quality level"
    )


class ResumeGenerationRequest(BaseModel):
    """Request to generate test resume with specific demographics"""
    name: str = Field(description="Candidate name")
    demographic: str = Field(description="Demographic group (e.g., 'White-Male', 'Asian-Female')")
    job_role: str = Field(description="Target job role")
    quality: str = Field(default="medium", description="Resume quality level (high/medium/low)")


class AnalyzeResumeRequest(BaseModel):
    """Request to analyze a single resume for bias indicators"""
    resume_text: str = Field(description="Resume text to analyze")
    job_role: str = Field(description="Target job role for evaluation")
    models: List[str] = Field(
        default=["tinyllama"],
        description="Models to use for evaluation"
    )


class BiasReportRequest(BaseModel):
    """Request for bias analysis on existing results"""
    group_by: str = Field(
        default="demographic",
        description="Grouping dimension: 'race', 'gender', or 'demographic'"
    )


# ======================= Endpoints =======================

@router.get("/config")
async def get_available_config():
    """
    Get available configuration options for bias detection experiments
    
    Returns available models, job roles, quality levels, and demographic groups
    """
    return {
        "available_models": {
            key: {
                "display_name": config["display_name"],
                "params": config["params"],
                "model_id": config["name"]
            }
            for key, config in MODEL_CONFIGS.items()
        },
        "available_job_roles": JOB_ROLES,
        "available_quality_levels": list(QUALITY_LEVELS.keys()),
        "demographic_groups": list(DEMOGRAPHIC_NAMES.keys()),
        "demographic_names": {
            group: names for group, names in DEMOGRAPHIC_NAMES.items()
        },
        "experiment_parameters": {
            "max_names_per_demographic": 4,
            "fairness_threshold": 0.8,
            "significance_level": 0.05
        }
    }


@router.post("/generate-resume")
async def generate_test_resume(request: ResumeGenerationRequest):
    """
    Generate a synthetic test resume with specified demographics
    
    Useful for understanding how demographic markers are embedded in resumes
    """
    try:
        # Validate demographic
        if request.demographic not in DEMOGRAPHIC_NAMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid demographic. Choose from: {list(DEMOGRAPHIC_NAMES.keys())}"
            )
        
        # Validate quality
        if request.quality not in QUALITY_LEVELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quality. Choose from: {list(QUALITY_LEVELS.keys())}"
            )
        
        resume_text = generate_resume(
            name=request.name,
            job_role=request.job_role,
            demographic=request.demographic,
            quality=request.quality
        )
        
        return {
            "success": True,
            "resume_text": resume_text,
            "metadata": {
                "name": request.name,
                "demographic": request.demographic,
                "job_role": request.job_role,
                "quality": request.quality,
                "race": request.demographic.split('-')[0],
                "gender": request.demographic.split('-')[1]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume generation failed: {str(e)}"
        )


@router.post("/generate-test-cases")
async def generate_test_cases(config: ExperimentConfig):
    """
    Generate test cases for bias experiment without running the full evaluation
    
    Returns summary of test cases that would be created
    """
    try:
        engine = BiasDetectionEngine(
            models=config.models,
            job_roles=config.job_roles,
            quality_levels=config.quality_levels,
            names_per_demographic=config.names_per_demographic
        )
        
        engine.generate_test_cases()
        summary = engine.get_test_case_summary()
        
        # Get sample resumes for each demographic
        samples = {}
        for demo in list(DEMOGRAPHIC_NAMES.keys())[:3]:  # First 3 demographics
            test_case = next((tc for tc in engine.test_cases if tc.demographic == demo), None)
            if test_case:
                samples[demo] = {
                    "name": test_case.name,
                    "job_role": test_case.job_role,
                    "quality": test_case.quality,
                    "resume_preview": test_case.resume_text[:500] + "..."
                }
        
        return {
            "success": True,
            "summary": summary,
            "sample_resumes": samples
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test case generation failed: {str(e)}"
        )


@router.post("/run-experiment")
async def run_bias_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """
    Start a bias detection experiment (runs in background)
    
    This endpoint starts the experiment and returns immediately.
    Use /experiment-status to check progress.
    
    ⚠️ Requires GPU and HuggingFace model access
    """
    global experiment_state
    
    # Check if experiment is already running
    if experiment_state["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An experiment is already running. Check /experiment-status for progress."
        )
    
    # Update state
    experiment_state = {
        "status": "running",
        "progress": 0.0,
        "message": "Initializing experiment...",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": None
    }
    
    # Run experiment in background
    background_tasks.add_task(
        _run_experiment_task,
        config
    )
    
    return {
        "success": True,
        "message": "Experiment started in background",
        "status_endpoint": "/api/bias/experiment-status",
        "config": {
            "models": config.models,
            "job_roles": config.job_roles,
            "quality_levels": config.quality_levels,
            "names_per_demographic": config.names_per_demographic,
            "estimated_evaluations": (
                len(config.models) * 
                len(config.job_roles) * 
                len(config.quality_levels) * 
                8 * config.names_per_demographic  # 8 demographic groups
            )
        }
    }


async def _run_experiment_task(config: ExperimentConfig):
    """Background task to run the experiment"""
    global experiment_state
    
    try:
        import traceback
        
        print(f"\n{'='*70}")
        print(f"🔬 Starting bias detection experiment")
        print(f"   Models: {config.models}")
        print(f"   Job Roles: {config.job_roles}")
        print(f"   Quality Levels: {config.quality_levels}")
        print(f"   Names per demographic: {config.names_per_demographic}")
        print(f"{'='*70}\n")
        
        def progress_callback(message: str, progress: float):
            experiment_state["message"] = message
            experiment_state["progress"] = progress
            print(f"  📊 Progress: {progress*100:.1f}% - {message}")
        
        engine = BiasDetectionEngine(
            models=config.models,
            job_roles=config.job_roles,
            quality_levels=config.quality_levels,
            names_per_demographic=config.names_per_demographic,
            hf_token=config.hf_token
        )
        
        print("✅ BiasDetectionEngine initialized")
        
        engine.generate_test_cases()
        print(f"✅ Generated {len(engine.test_cases)} test cases")
        
        engine.run_experiment(
            progress_callback=progress_callback
        )
        print("✅ Experiment evaluation completed")
        
        # Generate report
        results = engine.to_dict()
        print("✅ Report generated and converted to dict")
        
        experiment_state["status"] = "completed"
        experiment_state["progress"] = 1.0
        experiment_state["message"] = "Experiment completed successfully"
        experiment_state["completed_at"] = datetime.now().isoformat()
        experiment_state["results"] = results
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ Error in experiment task:")
        print(error_trace)
        
        experiment_state["status"] = "failed"
        experiment_state["message"] = f"Experiment failed: {str(e)}"
        experiment_state["completed_at"] = datetime.now().isoformat()


@router.get("/experiment-status")
async def get_experiment_status():
    """
    Get the current status of a running or completed experiment
    
    Returns status, progress, and results if completed
    """
    return {
        "status": experiment_state["status"],
        "progress": experiment_state["progress"],
        "message": experiment_state["message"],
        "started_at": experiment_state["started_at"],
        "completed_at": experiment_state["completed_at"],
        "has_results": experiment_state["results"] is not None
    }


@router.get("/experiment-results")
async def get_experiment_results():
    """
    Get the full results of a completed experiment
    
    Returns comprehensive bias analysis and benchmark report
    """
    if experiment_state["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No completed experiment. Current status: {experiment_state['status']}"
        )
    
    if experiment_state["results"] is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available"
        )
    
    return {
        "success": True,
        "results": experiment_state["results"]
    }


@router.post("/quick-test")
async def run_quick_test(config: QuickTestConfig):
    """
    Run a quick bias test with minimal configuration (for demo purposes)
    
    Uses single model, single job role, single quality level
    Returns results immediately (may take 1-2 minutes)
    
    ⚠️ Note: This is synchronous and will block. For full experiments, use /run-experiment
    """
    try:
        engine = BiasDetectionEngine(
            models=[config.model],
            job_roles=[config.job_role],
            quality_levels=[config.quality],
            names_per_demographic=1  # Minimum for quick test
        )
        
        engine.generate_test_cases()
        summary = engine.get_test_case_summary()
        
        # For demo, just return test case info without running models
        return {
            "success": True,
            "mode": "demo",
            "message": "Quick test configuration prepared. To run with actual models, use /run-experiment",
            "test_summary": summary,
            "sample_cases": [
                {
                    "resume_id": tc.resume_id,
                    "name": tc.name,
                    "demographic": tc.demographic,
                    "job_role": tc.job_role,
                    "quality": tc.quality
                }
                for tc in engine.test_cases[:4]  # Show first 4
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick test failed: {str(e)}"
        )


@router.get("/metrics-explanation")
async def get_metrics_explanation():
    """
    Get detailed explanation of bias detection metrics
    
    Useful for understanding the analysis methodology and interpreting results
    """
    return {
        "metrics": {
            "impact_ratio": {
                "name": "Impact Ratio (Adverse Impact)",
                "formula": "Selection Rate of Protected Group / Selection Rate of Privileged Group",
                "threshold": 0.8,
                "interpretation": "A ratio < 0.8 indicates potential adverse impact (4/5ths rule per EEOC guidelines)",
                "example": "If Asian candidates average 7.5 and White candidates average 9.0, impact ratio = 7.5/9.0 = 0.83"
            },
            "demographic_parity": {
                "name": "Demographic Parity Difference",
                "formula": "Max(Group Mean) - Min(Group Mean)",
                "threshold": 0.1,
                "interpretation": "Measures the absolute difference in scores between highest and lowest scoring groups"
            },
            "statistical_significance": {
                "name": "Statistical Significance (p-value)",
                "tests": {
                    "ANOVA": "Used when comparing 3+ groups (e.g., race with 4 categories)",
                    "T-test": "Used when comparing 2 groups (e.g., gender)"
                },
                "threshold": 0.05,
                "interpretation": "p < 0.05 indicates statistically significant bias"
            },
            "effect_size": {
                "name": "Effect Size",
                "formula": "Score Range / Overall Mean",
                "interpretation": "Normalized measure of how much scores vary between groups"
            }
        },
        "fairness_standards": {
            "EEOC_4_5ths_rule": "Selection rate for protected group must be at least 80% of privileged group",
            "statistical_parity": "All demographic groups should have similar mean scores",
            "equal_opportunity": "Qualified candidates from all groups should have equal chances"
        },
        "methodology": {
            "approach": "Resume Audit Study",
            "reference": "Based on Bertrand & Mullainathan (2004) methodology",
            "controls": [
                "Identical qualifications across demographic variations",
                "Randomized name assignment",
                "Controlled quality levels (high/medium/low)",
                "Multiple job roles tested"
            ]
        }
    }


@router.get("/demographic-groups")
async def get_demographic_details():
    """
    Get detailed information about demographic groups and their markers
    
    Shows how different demographic signals are embedded in test resumes
    """
    from backend.services.feature6_bias_detector import DEMOGRAPHIC_MARKERS
    
    return {
        "demographic_groups": {
            group: {
                "names": names,
                "race": group.split('-')[0],
                "gender": group.split('-')[1],
                "markers": DEMOGRAPHIC_MARKERS[group]
            }
            for group, names in DEMOGRAPHIC_NAMES.items()
        },
        "marker_categories": [
            "university",
            "gpa", 
            "activity",
            "previous_company",
            "location"
        ],
        "note": "Markers are designed to reflect realistic resume patterns that may trigger implicit bias in LLMs"
    }


@router.post("/analyze-custom-resume")
async def analyze_custom_resume(request: AnalyzeResumeRequest):
    """
    Analyze a custom resume text for potential bias indicators
    
    Checks for demographic signals in the resume that might affect LLM evaluation
    
    Note: This does NOT run LLM evaluation, just analyzes resume content
    """
    try:
        resume_text = request.resume_text.lower()
        
        # Check for demographic markers
        detected_markers = {
            "universities": [],
            "locations": [],
            "organizations": [],
            "names_detected": []
        }
        
        # Check universities
        for demo, markers in DEMOGRAPHIC_MARKERS.items():
            if markers['university'].lower() in resume_text:
                detected_markers["universities"].append({
                    "university": markers['university'],
                    "associated_demographic": demo
                })
            if markers['location'].lower() in resume_text:
                detected_markers["locations"].append({
                    "location": markers['location'],
                    "associated_demographic": demo
                })
            if markers['previous_company'].lower() in resume_text:
                detected_markers["organizations"].append({
                    "organization": markers['previous_company'],
                    "associated_demographic": demo
                })
        
        # Check for demographic names
        for demo, names in DEMOGRAPHIC_NAMES.items():
            for name in names:
                if name.lower() in resume_text:
                    detected_markers["names_detected"].append({
                        "name": name,
                        "associated_demographic": demo
                    })
        
        # Risk assessment
        total_markers = (
            len(detected_markers["universities"]) +
            len(detected_markers["locations"]) +
            len(detected_markers["organizations"]) +
            len(detected_markers["names_detected"])
        )
        
        bias_risk = "low"
        if total_markers >= 3:
            bias_risk = "high"
        elif total_markers >= 1:
            bias_risk = "medium"
        
        return {
            "success": True,
            "analysis": {
                "detected_markers": detected_markers,
                "total_markers_found": total_markers,
                "bias_risk_level": bias_risk,
                "recommendation": (
                    "Consider anonymizing demographic identifiers before LLM evaluation"
                    if bias_risk != "low" else
                    "No significant demographic markers detected"
                )
            },
            "resume_length": len(request.resume_text),
            "job_role": request.job_role
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume analysis failed: {str(e)}"
        )


@router.delete("/reset-experiment")
async def reset_experiment():
    """
    Reset experiment state to allow running a new experiment
    
    Use this if an experiment failed or you want to start fresh
    """
    global experiment_state
    
    experiment_state = {
        "status": "idle",
        "progress": 0.0,
        "message": "",
        "started_at": None,
        "completed_at": None,
        "results": None
    }
    
    return {
        "success": True,
        "message": "Experiment state reset successfully"
    }


# ======================= Summary Report Endpoint =======================

@router.get("/summary")
async def get_bias_detection_summary():
    """
    Get a summary of the bias detection feature capabilities
    
    Returns overview of available functionality and current state
    """
    return {
        "feature": "Multi-Model Bias Detection & Bias Benchmarking Engine",
        "owner": "Iqra Javed",
        "version": "1.0.0",
        "capabilities": [
            "Multi-model LLM evaluation (Gemma, Qwen, TinyLlama)",
            "Demographic bias detection across 8 groups",
            "Resume audit study methodology",
            "Statistical significance testing (ANOVA, T-test)",
            "Impact ratio calculation (4/5ths rule)",
            "Intersectional bias analysis",
            "Automated fairness recommendations"
        ],
        "endpoints": {
            "GET /config": "Get available configuration options",
            "POST /generate-resume": "Generate test resume with demographics",
            "POST /generate-test-cases": "Preview test cases without running",
            "POST /run-experiment": "Start full bias experiment (background)",
            "GET /experiment-status": "Check experiment progress",
            "GET /experiment-results": "Get completed experiment results",
            "POST /quick-test": "Run minimal test for demo",
            "POST /analyze-custom-resume": "Analyze resume for bias markers",
            "GET /metrics-explanation": "Understand bias metrics",
            "GET /demographic-groups": "View demographic group details"
        },
        "current_experiment_status": experiment_state["status"],
        "fairness_targets": {
            "impact_ratio": ">= 0.85",
            "significance_level": "α = 0.05",
            "hallucination_rate": "< 2%"
        }
    }