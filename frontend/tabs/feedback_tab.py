"""HR Feedback Tab - Submit NER Corrections"""
import streamlit as st
import requests


def render(api_base_url: str):
    """Render the HR Feedback tab"""
    st.header("HR Feedback & Corrections")
    st.markdown("Submit corrections to improve NER accuracy. System learns patterns automatically!")
    
    # Get feedback stats first
    try:
        stats_response = requests.get(f"{api_base_url}/api/feedback/stats", timeout=2)
        if stats_response.status_code == 200:
            feedback_data = stats_response.json()
            feedback_stats = feedback_data['stats']
            learned_patterns = feedback_data.get('learned_patterns', {})
        else:
            feedback_stats = None
            learned_patterns = {}
    except:
        feedback_stats = None
        learned_patterns = {}
    
    # Display stats if available
    if feedback_stats:
        st.markdown("---")
        st.subheader("Current Feedback Statistics")
        
        st.metric("Total Corrections", feedback_stats['total_corrections'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name Corrections", feedback_stats['corrections_by_field'].get('name', 0))
        with col2:
            st.metric("Email Corrections", feedback_stats['corrections_by_field'].get('email', 0))
        with col3:
            st.metric("Phone Corrections", feedback_stats['corrections_by_field'].get('phone', 0))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Name Patterns", feedback_stats['patterns_learned'].get('name', 0))
        with col2:
            st.metric("Email Patterns", feedback_stats['patterns_learned'].get('email', 0))
        with col3:
            st.metric("Phone Patterns", feedback_stats['patterns_learned'].get('phone', 0))
    
    st.markdown("---")
    
    # Two modes: Select from uploaded resumes or manual entry
    feedback_mode = st.radio(
        "Feedback Mode",
        ["Review Uploaded Resumes", "Manual Entry"],
        horizontal=True
    )
    
    if feedback_mode == "Review Uploaded Resumes":
        if not st.session_state.resumes_data:
            st.warning("No resumes uploaded yet. Please upload resumes in the 'Resumes' tab first.")
        else:
            st.subheader("Review & Correct Resume Extractions")
            
            # Select resume to review
            selected_idx = st.selectbox(
                "Select Resume to Review",
                options=range(len(st.session_state.resumes_data)),
                format_func=lambda x: f"{st.session_state.resumes_data[x]['filename']} - {st.session_state.resumes_data[x]['data'].get('name', 'Unknown')}"
            )
            
            selected_resume = st.session_state.resumes_data[selected_idx]
            resume_data = selected_resume['data']
            
            st.markdown(f"**File:** {selected_resume['filename']}")
            st.markdown(f"**Resume ID:** `{resume_data.get('resume_id', 'N/A')}`")
            
            st.markdown("---")
            
            # Show current extractions and allow corrections
            st.subheader("Current Extractions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Name")
                current_name = resume_data.get('name', 'Unknown')
                st.info(f"Extracted: **{current_name}**")
                
                correct_name = st.text_input(
                    "Correct Name (leave empty if correct)",
                    placeholder="e.g., Dr. María González",
                    key=f"name_{selected_idx}"
                )
                
                st.markdown("### Email")
                current_email = resume_data.get('email', 'unknown@email.com')
                st.info(f"Extracted: **{current_email}**")
                
                correct_email = st.text_input(
                    "Correct Email (leave empty if correct)",
                    placeholder="e.g., john.doe@company.com",
                    key=f"email_{selected_idx}"
                )
            
            with col2:
                st.markdown("### Phone")
                current_phone = resume_data.get('phone', '') or 'Not found'
                st.info(f"Extracted: **{current_phone}**")
                
                correct_phone = st.text_input(
                    "Correct Phone (leave empty if correct)",
                    placeholder="e.g., +1 (555) 123-4567",
                    key=f"phone_{selected_idx}"
                )
            
            st.markdown("---")
            
            # Submit corrections
            if st.button("Submit Corrections", type="primary", key=f"submit_{selected_idx}"):
                corrections_submitted = []
                
                # Check which fields need correction
                if correct_name and correct_name != current_name:
                    corrections_submitted.append({
                        'field': 'name',
                        'extracted': current_name,
                        'correct': correct_name
                    })
                
                if correct_email and correct_email != current_email:
                    corrections_submitted.append({
                        'field': 'email',
                        'extracted': current_email,
                        'correct': correct_email
                    })
                
                if correct_phone and correct_phone != current_phone:
                    corrections_submitted.append({
                        'field': 'phone',
                        'extracted': current_phone,
                        'correct': correct_phone
                    })
                
                if not corrections_submitted:
                    st.info("No corrections to submit. All fields are correct!")
                else:
                    with st.spinner("Submitting corrections..."):
                        success_count = 0
                        results = []
                        
                        for correction in corrections_submitted:
                            try:
                                response = requests.post(
                                    f"{api_base_url}/api/feedback/ner-correction",
                                    json={
                                        "resume_id": resume_data.get('resume_id', 'unknown'),
                                        "field": correction['field'],
                                        "extracted": correction['extracted'],
                                        "correct": correction['correct'],
                                        "resume_text": resume_data.get('text', '')
                                    }
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    results.append(result)
                                    success_count += 1
                                    
                                    # Update the resume data immediately
                                    if correction['field'] == 'name':
                                        resume_data['name'] = correction['correct']
                                    elif correction['field'] == 'email':
                                        resume_data['email'] = correction['correct']
                                    elif correction['field'] == 'phone':
                                        resume_data['phone'] = correction['correct']
                                else:
                                    st.error(f"Failed to submit {correction['field']}: {response.text}")
                            
                            except Exception as e:
                                st.error(f"Error submitting {correction['field']}: {str(e)}")
                        
                        if success_count > 0:
                            st.success(f"Successfully submitted {success_count} correction(s)!")
                            
                            # Show what was learned
                            for result in results:
                                if result.get('learned_pattern'):
                                    st.info(f"Learned new pattern for {result['field']}!")
                            
                            # Update session state
                            st.session_state.resumes_data[selected_idx]['data'] = resume_data
                            
                            st.balloons()
                            st.rerun()
    
    else:  # Manual Entry Mode
        st.subheader("Manual Correction Entry")
        st.markdown("Manually enter corrections for NER improvements")
        
        with st.form("manual_correction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                manual_field = st.selectbox(
                    "Field to Correct",
                    ["name", "email", "phone"]
                )
                
                manual_extracted = st.text_input(
                    "What was Extracted (wrong)",
                    placeholder="e.g., Unknown"
                )
                
                manual_correct = st.text_input(
                    "Correct Value",
                    placeholder="e.g., Dr. María González"
                )
            
            with col2:
                manual_resume_text = st.text_area(
                    "Resume Text Snippet",
                    height=150,
                    placeholder="Paste a snippet of the resume containing the correct value..."
                )
                
                manual_resume_id = st.text_input(
                    "Resume ID (optional)",
                    placeholder="e.g., RES_001"
                )
            
            submitted = st.form_submit_button("Submit Manual Correction", type="primary")
            
            if submitted:
                if not manual_extracted or not manual_correct or not manual_resume_text:
                    st.error("Please fill all required fields")
                else:
                    with st.spinner("Submitting correction..."):
                        try:
                            response = requests.post(
                                f"{api_base_url}/api/feedback/ner-correction",
                                json={
                                    "resume_id": manual_resume_id or "manual_entry",
                                    "field": manual_field,
                                    "extracted": manual_extracted,
                                    "correct": manual_correct,
                                    "resume_text": manual_resume_text
                                }
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("Correction submitted successfully!")
                                
                                if result.get('learned_pattern'):
                                    st.info(f"Learned new pattern: {result['learned_pattern'][:80]}...")
                                
                                st.info(f"Total {result['field']} corrections: {result['correction_count']}")
                                st.info(f"Total {result['field']} patterns: {result['total_patterns']}")
                                
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Failed: {response.text}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    # Show learned patterns
    st.markdown("---")
    st.subheader("Learned Patterns")
    
    if learned_patterns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("Name Patterns")
            name_patterns = learned_patterns.get('name_patterns', [])
            if name_patterns:
                for i, pattern in enumerate(name_patterns[:5], 1):
                    with st.expander(f"Pattern {i}"):
                        st.code(pattern, language="regex")
            else:
                st.info("No patterns yet")
        
        with col2:
            st.markdown("Email Patterns")
            email_patterns = learned_patterns.get('email_patterns', [])
            if email_patterns:
                for i, pattern in enumerate(email_patterns[:5], 1):
                    with st.expander(f"Pattern {i}"):
                        st.code(pattern, language="regex")
            else:
                st.info("No patterns yet")
        
        with col3:
            st.markdown("Phone Patterns")
            phone_patterns = learned_patterns.get('phone_patterns', [])
            if phone_patterns:
                for i, pattern in enumerate(phone_patterns[:5], 1):
                    with st.expander(f"Pattern {i}"):
                        st.code(pattern, language="regex")
            else:
                st.info("No patterns yet")
    else:
        st.info("No patterns learned yet. Submit corrections to start learning!")
    
    # Test patterns
    st.markdown("---")
    st.subheader("Test Learned Patterns")
    
    with st.form("test_pattern_form"):
        test_field = st.selectbox("Field to Test", ["name", "email", "phone"])
        test_text = st.text_area(
            "Test Text",
            placeholder="Enter text to test pattern matching...",
            height=100
        )
        
        test_submitted = st.form_submit_button("Test Patterns")
        
        if test_submitted and test_text:
            try:
                response = requests.post(
                    f"{api_base_url}/api/feedback/test-pattern",
                    params={"field": test_field, "text": test_text}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['matches']:
                        st.success(f"Found {result['match_count']} match(es)!")
                        for match in result['matches']:
                            st.code(match)
                    else:
                        st.info(f"No matches found using {result['patterns_used']} pattern(s)")
                else:
                    st.error("Failed to test patterns")
            except Exception as e:
                st.error(f"Error: {str(e)}")
