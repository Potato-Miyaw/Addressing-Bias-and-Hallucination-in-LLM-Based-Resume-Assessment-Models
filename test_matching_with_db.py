"""
Test Matching with Database Job Selection
Verify that matching works with job_id from database
"""

import asyncio
import requests
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from backend.database import (
    connect_to_mongo,
    close_mongo_connection,
    save_job_description,
    list_job_descriptions
)
from backend.utils.id_generator import generate_job_id

API_URL = "http://localhost:8000"

async def test_matching_with_job_selection():
    """Test the new job selection feature"""
    
    print("="*80)
    print("🧪 Testing Matching with Database Job Selection")
    print("="*80)
    
    try:
        # Step 1: Connect to database
        print("\n1️⃣ Connecting to MongoDB...")
        await connect_to_mongo()
        print("   ✅ Connected")
        
        # Step 2: Ensure we have a test job in database
        print("\n2️⃣ Creating test job in database...")
        test_jd_text = """
        Senior Data Scientist Position
        
        Requirements:
        - 5+ years of experience in machine learning
        - Strong Python and SQL skills
        - Experience with TensorFlow and PyTorch
        - Master's degree in Computer Science
        """
        
        job_id, content_hash = generate_job_id(test_jd_text, "Senior Data Scientist")
        
        job_data = {
            "job_id": job_id,
            "content_hash": content_hash,
            "job_title": "Senior Data Scientist",
            "jd_text": test_jd_text,
            "required_skills": ["Python", "SQL", "TensorFlow", "PyTorch"],
            "required_experience": 5,
            "required_education": "Master's degree",
            "certifications": [],
            "extraction_status": "SUCCESS"
        }
        
        await save_job_description(job_data)
        print(f"   ✅ Test job saved: {job_id}")
        
        # Step 3: List jobs from database
        print("\n3️⃣ Fetching jobs from database...")
        jobs = await list_job_descriptions(limit=5)
        print(f"   ✅ Found {len(jobs)} jobs")
        for job in jobs:
            print(f"      - {job['job_title']} ({job['job_id'][:16]}...)")
        
        # Step 4: Test API endpoint to list jobs
        print("\n4️⃣ Testing API endpoint: GET /api/data/jobs")
        response = requests.get(f"{API_URL}/api/data/jobs?limit=10")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API returned {data['count']} jobs")
            print(f"      Total in DB: {data['total']}")
        else:
            print(f"   ❌ API request failed: {response.status_code}")
            print(f"      Make sure the server is running: python backend/app.py")
            return
        
        # Step 5: Test matching with job_id
        print("\n5️⃣ Testing matching endpoint: POST /api/match/with-job-id")
        
        # Sample resume data
        test_resume = {
            "resume_id": "test_resume_001",
            "text": """
            John Doe
            Email: john@email.com
            
            EXPERIENCE:
            Senior ML Engineer at TechCorp (5 years)
            - Developed ML models using Python and TensorFlow
            - Worked with large-scale SQL databases
            
            SKILLS:
            Python, TensorFlow, SQL, PyTorch, Machine Learning
            
            EDUCATION:
            Master of Science in Computer Science, Stanford University
            """,
            "skills": ["Python", "TensorFlow", "SQL", "PyTorch", "Machine Learning"],
            "experience": {"years": 5},
            "education": ["Master's in Computer Science"]
        }
        
        match_response = requests.post(
            f"{API_URL}/api/match/with-job-id",
            json={
                "resume_id": test_resume["resume_id"],
                "job_id": job_id,
                "resume_data": test_resume
            }
        )
        
        if match_response.status_code == 200:
            match_result = match_response.json()
            print(f"   ✅ Matching successful!")
            print(f"      Job: {match_result['job_title']}")
            print(f"      Match Score: {match_result['match_result']['match_score']:.1f}%")
            print(f"      Skill Match: {match_result['match_result']['skill_match']:.1f}%")
            print(f"      Experience Match: {match_result['match_result']['experience_match']:.1f}%")
            
            if match_result['match_result']['skill_gaps']:
                print(f"      Skill Gaps: {', '.join(match_result['match_result']['skill_gaps'])}")
            else:
                print(f"      Skill Gaps: None ✅")
        else:
            print(f"   ❌ Matching failed: {match_response.status_code}")
            print(f"      {match_response.text}")
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("="*80)
        print("\n💡 Next Steps:")
        print("   1. Start the FastAPI server: python backend/app.py")
        print("   2. Start Streamlit: streamlit run frontend/streamlit_app.py")
        print("   3. Go to Matching tab and select a job from database")
        print("   4. Upload resumes and match!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔌 Closing MongoDB connection...")
        await close_mongo_connection()

if __name__ == "__main__":
    print("\n🚀 Starting matching integration test...\n")
    print("⚠️  Prerequisites:")
    print("   1. MongoDB running on mongodb://localhost:27017/")
    print("   2. FastAPI server running on http://localhost:8000")
    print("      (Start with: python backend/app.py)\n")
    
    asyncio.run(test_matching_with_job_selection())
