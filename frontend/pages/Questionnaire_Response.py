"""
Candidate Questionnaire Response Page
Candidates access this via token link sent by HR
"""

import streamlit as st
import requests
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Questionnaire - Candidate Response",
    layout="wide"
)

# Get token from URL parameters
query_params = st.query_params
token = query_params.get("token", None)

# Header
st.title("Pre-Interview Questionnaire")
st.markdown("Please answer the following questions to help us better understand your qualifications")

if not token:
    st.error("Invalid or missing invitation link")
    st.info("Please use the invitation link sent to your email")
    st.stop()

# Validate token and get questionnaire
try:
    with st.spinner("Loading questionnaire..."):
        response = requests.get(f"{API_BASE_URL}/api/questionnaire/validate-token/{token}")
    
    if response.status_code == 200:
        data = response.json()
        
        if not data.get('success'):
            # Token invalid or already used
            error_type = data.get('error')
            
            if error_type == 'already_submitted':
                st.success("You have already submitted this questionnaire")
                st.info(f"Submitted on: {data['submitted_at']}")
                st.markdown("---")
                st.markdown("Thank you for your response! The HR team will review your answers and contact you soon.")
            elif error_type == 'expired':
                st.error("This invitation link has expired")
                st.info(f"Expired on: {data['expired_at']}")
                st.markdown("Please contact HR for a new invitation link.")
            else:
                st.error(f"{data.get('message', 'Invalid invitation')}")
            
            st.stop()
        
        # Valid token - show questionnaire
        invitation = data['invitation']
        questionnaire = data['questionnaire']
        
        # Welcome message
        st.success(f"Welcome, **{invitation['candidate_name']}**!")
        st.info(f"{invitation['candidate_email']} | Expires: {invitation['expires_at'][:10]}")
        
        st.markdown("---")
        
        # Questionnaire details
        st.header(questionnaire['title'])
        st.markdown(questionnaire['description'])
        
        st.markdown("---")
        
        # Display questions and collect answers
        st.markdown(f"### Questions ({questionnaire['total_questions']} total)")
        
        answers = []
        
        with st.form("questionnaire_form"):
            for i, question in enumerate(questionnaire['questions'], 1):
                st.markdown(f"#### Question {i}")
                st.markdown(f"**{question['text']}**")
                
                if not question['required']:
                    st.caption("(Optional)")
                
                answer_value = None
                
                # Render based on question type
                if question['type'] == 'text':
                    answer_value = st.text_area(
                        f"Your answer:",
                        key=f"q_{question['question_id']}",
                        height=100,
                        help="Please provide a detailed answer"
                    )
                
                elif question['type'] == 'multiple_choice':
                    answer_value = st.radio(
                        "Select one:",
                        options=question.get('options', []),
                        key=f"q_{question['question_id']}"
                    )
                
                elif question['type'] == 'rating':
                    answer_value = st.select_slider(
                        "Rate:",
                        options=question.get('options', ['1', '2', '3', '4', '5']),
                        key=f"q_{question['question_id']}"
                    )
                
                elif question['type'] == 'number':
                    answer_value = st.number_input(
                        "Enter number:",
                        min_value=0,
                        max_value=100,
                        key=f"q_{question['question_id']}"
                    )
                
                # Store answer
                answers.append({
                    "question_id": question['question_id'],
                    "question_text": question['text'],
                    "question_type": question['type'],
                    "answer": answer_value,
                    "required": question['required']
                })
                
                st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button("Submit Questionnaire", type="primary", use_container_width=True)
            
            if submitted:
                # Validate required questions
                missing_required = []
                for ans in answers:
                    if ans['required'] and (ans['answer'] is None or ans['answer'] == ''):
                        missing_required.append(ans['question_id'])
                
                if missing_required:
                    st.error(f"Please answer all required questions ({len(missing_required)} missing)")
                else:
                    # Submit responses
                    with st.spinner("Submitting your answers..."):
                        submit_response = requests.post(
                            f"{API_BASE_URL}/api/questionnaire/submit",
                            json={
                                "token": token,
                                "answers": answers
                            }
                        )
                    
                    if submit_response.status_code == 200:
                        result = submit_response.json()
                        st.success(result['message'])
                        st.balloons()
                        
                        st.markdown("---")
                        st.markdown("### Thank you for completing the questionnaire!")
                        st.info("The HR team will review your responses and contact you regarding the next steps in the hiring process.")
                        
                        # Disable further submissions
                        st.stop()
                    else:
                        error_msg = submit_response.json().get('detail', 'Submission failed')
                        st.error(f"{error_msg}")
    
    else:
        st.error("Failed to validate invitation link")
        st.info("Please contact HR if you continue to experience issues")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please contact HR for assistance")
