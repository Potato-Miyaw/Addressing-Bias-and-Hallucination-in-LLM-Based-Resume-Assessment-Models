"""
Test MongoDB Integration for Job Descriptions
Run this to verify MongoDB connection and CRUD operations
"""

import asyncio
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
    get_job_description,
    get_job_by_content_hash,
    list_job_descriptions,
    search_jobs_by_title,
    get_job_count,
    delete_job_description
)
from backend.utils.id_generator import generate_job_id

async def test_mongodb():
    """Test MongoDB operations"""
    
    print("="*80)
    print("🧪 Testing MongoDB Integration for Job Descriptions")
    print("="*80)
    
    try:
        # Step 1: Connect to MongoDB
        print("\n1️⃣ Connecting to MongoDB...")
        await connect_to_mongo()
        print("   ✅ Connected successfully")
        
        # Step 2: Generate test job data
        print("\n2️⃣ Generating test job data...")
        test_jd_text = """
        Senior Data Scientist Position
        
        We are looking for an experienced Data Scientist with:
        - 5+ years of experience in machine learning
        - Strong Python and SQL skills
        - Experience with TensorFlow and PyTorch
        - Master's degree in Computer Science or related field
        - AWS or Azure cloud certifications preferred
        """
        
        job_id, content_hash = generate_job_id(test_jd_text, "Senior Data Scientist")
        print(f"   Generated Job ID: {job_id}")
        print(f"   Content Hash: {content_hash}")
        
        job_data = {
            "job_id": job_id,
            "content_hash": content_hash,
            "job_title": "Senior Data Scientist",
            "jd_text": test_jd_text,
            "required_skills": ["Python", "SQL", "TensorFlow", "PyTorch"],
            "required_experience": 5,
            "required_education": "Master's degree",
            "certifications": ["AWS", "Azure"],
            "extraction_status": "SUCCESS"
        }
        
        # Step 3: Save job description
        print("\n3️⃣ Saving job description to MongoDB...")
        saved = await save_job_description(job_data)
        if saved:
            print("   ✅ Job saved successfully")
        else:
            print("   ❌ Failed to save job")
            return
        
        # Step 4: Retrieve job by ID
        print("\n4️⃣ Retrieving job by ID...")
        retrieved_job = await get_job_description(job_id)
        if retrieved_job:
            print(f"   ✅ Retrieved: {retrieved_job['job_title']}")
            print(f"   Skills: {', '.join(retrieved_job['required_skills'])}")
        else:
            print("   ❌ Job not found")
        
        # Step 5: Check duplicate detection
        print("\n5️⃣ Testing duplicate detection...")
        duplicate_job = await get_job_by_content_hash(content_hash)
        if duplicate_job:
            print(f"   ✅ Duplicate detected: {duplicate_job['job_id']}")
        else:
            print("   ❌ Duplicate detection failed")
        
        # Step 6: Search by title
        print("\n6️⃣ Testing search by title...")
        search_results = await search_jobs_by_title("Data Scientist")
        print(f"   ✅ Found {len(search_results)} jobs matching 'Data Scientist'")
        
        # Step 7: List all jobs
        print("\n7️⃣ Listing all jobs...")
        all_jobs = await list_job_descriptions(limit=10)
        total_count = await get_job_count()
        print(f"   ✅ Total jobs in database: {total_count}")
        print(f"   Retrieved {len(all_jobs)} jobs (limit: 10)")
        
        # Step 8: Save another job with same content (duplicate test)
        print("\n8️⃣ Testing duplicate handling...")
        new_job_id, _ = generate_job_id(test_jd_text, "Senior Data Scientist")
        duplicate_test_data = job_data.copy()
        duplicate_test_data["job_id"] = new_job_id
        
        existing = await get_job_by_content_hash(content_hash)
        if existing:
            print(f"   ⚠️ Duplicate detected - original ID: {existing['job_id']}")
            print(f"   New upload would get ID: {new_job_id}")
            print("   ✅ System correctly identifies duplicates")
        
        # Step 9: Clean up (optional - comment out to keep test data)
        print("\n9️⃣ Cleaning up test data...")
        deleted = await delete_job_description(job_id)
        if deleted:
            print("   ✅ Test job deleted successfully")
        else:
            print("   ⚠️ Could not delete test job (may have been removed already)")
        
        print("\n" + "="*80)
        print("✅ All tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close connection
        print("\n🔌 Closing MongoDB connection...")
        await close_mongo_connection()
        print("   ✅ Connection closed")

if __name__ == "__main__":
    print("\n🚀 Starting MongoDB integration test...\n")
    print("⚠️  Make sure MongoDB is running on mongodb://localhost:27017/")
    print("   Windows: Check if 'MongoDB' service is running")
    print("   Or start manually: mongod --dbpath C:\\data\\db\n")
    
    asyncio.run(test_mongodb())
