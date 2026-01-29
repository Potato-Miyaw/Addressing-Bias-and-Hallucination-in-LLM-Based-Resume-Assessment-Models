"""Questionnaire Tab for HR Portal - Create and send questionnaires to candidates"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime


def render(api_base_url: str):
    """Render the Questionnaire Management tab"""
    st.header("📋 Questionnaire Management")
    st.markdown("Create and send personalized questionnaires to candidates based on match results")
    
    # Sub-tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🎯 Generate & Send", "📊 View Questionnaires", "📥 View Responses"])
    
    with tab1:
        render_generate_send(api_base_url)
    
    with tab2:
        render_view_questionnaires(api_base_url)
    
    with tab3:
        render_view_responses(api_base_url)


def render_generate_send(api_base_url: str):
    """Tab 1: Generate and send questionnaires"""
    st.subheader("🎯 Generate & Send Questionnaire")
    st.markdown("Select a candidate from matching results to auto-generate a personalized questionnaire")
    
    # Check if we have ranked candidates in session
    if hasattr(st.session_state, 'ranked_candidates') and st.session_state.ranked_candidates:
        st.success(f"✅ Found {len(st.session_state.ranked_candidates)} ranked candidates from matching")
        
        # Candidate selection
        candidate_options = {}
        for candidate in st.session_state.ranked_candidates:
            # Try multiple sources for name and email (match DB vs resume_data)
            name = (
                candidate.get('candidate_name') or 
                candidate.get('resume_data', {}).get('name') or 
                'Unknown'
            )
            email = (
                candidate.get('candidate_email') or 
                candidate.get('resume_data', {}).get('email') or 
                'No email'
            )
            rank = candidate['rank']
            match_score = candidate['match_data']['match_score']
            
            label = f"Rank #{rank}: {name} ({email}) - Match: {match_score:.1f}%"
            candidate_options[label] = candidate
        
        selected_candidate_key = st.selectbox(
            "Select Candidate:",
            options=list(candidate_options.keys()),
            key="questionnaire_candidate_selector"
        )
        
        selected_candidate = candidate_options[selected_candidate_key]
        
        # Show candidate details
        with st.expander("👤 Candidate Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Personal Info:**")
                
                # Get original values - check both match DB fields and resume_data
                original_name = (
                    selected_candidate.get('candidate_name') or 
                    selected_candidate.get('resume_data', {}).get('name') or 
                    'Unknown'
                )
                original_email = (
                    selected_candidate.get('candidate_email') or 
                    selected_candidate.get('resume_data', {}).get('email') or 
                    ''
                )
                
                # Check if empty/invalid
                name_is_empty = not original_name or original_name == 'Unknown'
                email_is_empty = not original_email or original_email == 'N/A'
                
                # Editable fields
                candidate_name = st.text_input(
                    "Candidate Name:",
                    value=original_name if not name_is_empty else "",
                    placeholder="Enter candidate name",
                    help="✏️ You can edit this if name is missing or incorrect",
                    key="edit_candidate_name"
                )
                
                candidate_email = st.text_input(
                    "Candidate Email:",
                    value=original_email if not email_is_empty else "",
                    placeholder="candidate@example.com",
                    help="✏️ You can edit this if email is missing or incorrect",
                    key="edit_candidate_email"
                )
                
                # Get candidate phone number (with fallback)
                candidate_phone = (
                    candidate.get('candidate_phone') or 
                    candidate.get('resume_data', {}).get('phone') or 
                    ""
                )
                
                candidate_phone = st.text_input(
                    "📱 Candidate Phone (for WhatsApp):",
                    value=candidate_phone,
                    placeholder="+1234567890 (E.164 format)",
                    help="✏️ Phone number for WhatsApp delivery (format: +country_code + number)",
                    key="edit_candidate_phone"
                )
                
                # Show warning if fields are empty
                if not candidate_name or not candidate_email:
                    st.warning("⚠️ Name and email are required to send questionnaire")
                
                st.write(f"**Resume ID:** `{selected_candidate['resume_id'][:12]}...`")
            
            with col2:
                st.markdown("**Match Details:**")
                st.write(f"**Rank:** #{selected_candidate['rank']}")
                st.write(f"**Match Score:** {selected_candidate['match_data']['match_score']:.1f}%")
                st.write(f"**Skill Match:** {selected_candidate['match_data']['skill_match']:.1f}%")
            
            # Skill gaps
            if selected_candidate['match_data']['skill_gaps']:
                st.markdown("**🔴 Skill Gaps (will be questioned):**")
                for gap in selected_candidate['match_data']['skill_gaps'][:5]:
                    st.write(f"  • {gap}")
        
        st.markdown("---")
        
        # Generation options
        st.subheader("⚙️ Generation Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_ai = st.checkbox(
                "Use AI (SmolLM2) for behavioral questions",
                value=True,
                help="AI generates intelligent behavioral questions. Uncheck for template-only."
            )
        
        with col2:
            expires_in_days = st.slider(
                "Link expires in (days):",
                min_value=1,
                max_value=30,
                value=7,
                help="How long the invitation link remains valid"
            )
        
        with col3:
            delivery_method = st.radio(
                "📬 Delivery Method:",
                options=["email", "whatsapp", "both"],
                index=0,
                help="Choose how to send the invitation link"
            )
        
        # Check if questionnaire was just generated - show send button
        if hasattr(st.session_state, 'generated_questionnaire') and st.session_state.generated_questionnaire:
            questionnaire = st.session_state.generated_questionnaire
            
            st.success(f"✅ Questionnaire generated with {questionnaire['total_questions']} questions")
            
            # Show generated questions
            with st.expander("📝 Generated Questions", expanded=True):
                for i, q in enumerate(questionnaire['questions'], 1):
                    st.markdown(f"**Q{i}:** {q['text']}")
                    if q.get('type') in ['multiple_choice', 'rating']:
                        st.write(f"   Options: {', '.join(q.get('options', []))}")
                    if q.get('generated_by'):
                        st.caption(f"   🤖 {q['generated_by']}")
                    st.markdown("")
            
            # Send invitation section
            st.markdown("---")
            st.subheader("📧 Send Invitation")
            
            # Show delivery info based on method
            if delivery_method == "email":
                st.info(f"**Sending to:** {candidate_name} ({candidate_email}) via Email")
            elif delivery_method == "whatsapp":
                st.info(f"**Sending to:** {candidate_name} ({candidate_phone}) via WhatsApp")
            else:
                st.info(f"**Sending to:** {candidate_name} via Email ({candidate_email}) and WhatsApp ({candidate_phone})")
            
            col_send1, col_send2 = st.columns([1, 3])
            with col_send1:
                if st.button("📤 Send Now", type="primary", key="send_invite_btn"):
                    # Validate based on delivery method
                    validation_error = None
                    if delivery_method in ["email", "both"] and not candidate_email:
                        validation_error = "Email is required for email delivery"
                    elif delivery_method in ["whatsapp", "both"] and not candidate_phone:
                        validation_error = "Phone number is required for WhatsApp delivery"
                    
                    if validation_error:
                        st.error(f"❌ {validation_error}")
                    else:
                        with st.spinner("Sending invitation..."):
                            job_id = getattr(st.session_state, 'rank_job_id', None)
                            resume_id = selected_candidate['resume_id']
                            
                            invite_payload = {
                                "questionnaire_id": questionnaire['questionnaire_id'],
                                "candidate_email": candidate_email,
                                "candidate_name": candidate_name,
                                "job_id": job_id,
                                "resume_id": resume_id,
                                "expires_in_days": expires_in_days,
                                "delivery_method": delivery_method
                            }
                            
                            # Add phone if using WhatsApp
                            if delivery_method in ["whatsapp", "both"]:
                                invite_payload["candidate_phone"] = candidate_phone
                            
                            invite_response = requests.post(
                                f"{api_base_url}/api/questionnaire/invite",
                                json=invite_payload
                            )
                            
                            if invite_response.status_code == 200:
                                invite_result = invite_response.json()
                                email_sent = invite_result.get('email_sent', False)
                                whatsapp_sent = invite_result.get('whatsapp_sent', False)
                            
                                # Store invitation result + details for potential resend
                                st.session_state.invitation_sent = True
                                st.session_state.invitation_link = invite_result['invitation_link']
                                st.session_state.invitation_token = invite_result['token']
                                st.session_state.questionnaire_title = questionnaire.get('title', 'Position Questionnaire')
                                st.session_state.email_sent = email_sent
                                st.session_state.whatsapp_sent = whatsapp_sent
                                st.session_state.delivery_method = delivery_method
                                st.session_state.expires_at = invite_result['expires_at']
                                st.session_state.candidate_email_for_resend = candidate_email
                                st.session_state.candidate_name_for_resend = candidate_name
                                st.session_state.candidate_phone_for_resend = candidate_phone
                            
                                # Clear questionnaire from session
                                del st.session_state.generated_questionnaire
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to send invitation: {invite_response.text}")
            
            with col_send2:
                if st.button("❌ Cancel", key="cancel_send"):
                    del st.session_state.generated_questionnaire
                    st.rerun()
        
        # Show invitation result if just sent
        elif hasattr(st.session_state, 'invitation_sent') and st.session_state.invitation_sent:
            email_sent = st.session_state.get('email_sent', False)
            whatsapp_sent = st.session_state.get('whatsapp_sent', False)
            delivery_method = st.session_state.get('delivery_method', 'email')
            
            # Show delivery status
            if delivery_method == "both":
                if email_sent and whatsapp_sent:
                    st.success(f"✅ Invitation sent successfully via Email and WhatsApp!")
                elif email_sent:
                    st.warning(f"⚠️ Email sent successfully, but WhatsApp failed. Check Twilio configuration.")
                elif whatsapp_sent:
                    st.warning(f"⚠️ WhatsApp sent successfully, but Email failed. Check SMTP configuration.")
                else:
                    st.error(f"❌ Both Email and WhatsApp failed to send. Copy the link manually.")
            elif delivery_method == "email":
                if email_sent:
                    st.success(f"✅ Invitation sent successfully via Email!")
                else:
                    st.warning(f"⚠️ Email failed to send. Check SMTP configuration or copy the link manually.")
            elif delivery_method == "whatsapp":
                if whatsapp_sent:
                    st.success(f"✅ Invitation sent successfully via WhatsApp!")
                else:
                    st.warning(f"⚠️ WhatsApp failed to send. Check Twilio configuration or copy the link manually.")
            
            # Show invitation details
            st.info(f"**Invitation Link:** {st.session_state.invitation_link}")
            st.caption(f"Expires: {st.session_state.expires_at}")
            
            # Copy link button
            st.code(st.session_state.invitation_link, language=None)
            
            # Show retry options if something failed
            if not email_sent or not whatsapp_sent:
                st.info("💡 **Tip:** Configure SMTP (email) and Twilio (WhatsApp) settings in `.env` file. See `.env.example` and `EMAIL_SETUP.md` for instructions.")
                
                # Resend buttons
                col_resend1, col_resend2, col_resend3 = st.columns(3)
                
                if delivery_method in ["email", "both"] and not email_sent:
                    with col_resend1:
                        if st.button("🔄 Retry Email", key="resend_email_btn", type="secondary"):
                            with st.spinner("Retrying email..."):
                                resend_response = requests.post(
                                    f"{api_base_url}/api/questionnaire/resend-email",
                                    json={
                                        "token": st.session_state.invitation_token,
                                        "candidate_email": st.session_state.candidate_email_for_resend,
                                        "candidate_name": st.session_state.candidate_name_for_resend,
                                        "invitation_link": st.session_state.invitation_link,
                                        "job_title": st.session_state.questionnaire_title,
                                        "expires_at": st.session_state.expires_at
                                    }
                                )
                                
                                if resend_response.status_code == 200:
                                    result = resend_response.json()
                                    if result.get('success'):
                                        st.session_state.email_sent = True
                                        st.success("✅ Email sent successfully!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Email still failed. Check SMTP configuration.")
                                else:
                                    st.error(f"❌ Resend failed: {resend_response.text}")
                
                if delivery_method in ["whatsapp", "both"] and not whatsapp_sent:
                    with col_resend2:
                        if st.button("🔄 Retry WhatsApp", key="resend_whatsapp_btn", type="secondary"):
                            st.info("💡 WhatsApp retry coming soon! For now, copy the link and send manually.")
            
            # Clear button
            if st.button("✅ Done - Generate Another", key="clear_result"):
                # Clean up all session state
                for key in ['invitation_sent', 'invitation_link', 'invitation_token', 
                           'questionnaire_title', 'email_sent', 'whatsapp_sent', 'delivery_method', 
                           'expires_at', 'candidate_email_for_resend', 'candidate_name_for_resend',
                           'candidate_phone_for_resend']:
                    if hasattr(st.session_state, key):
                        delattr(st.session_state, key)
                st.rerun()
        
        # Generate button (only show if no questionnaire in progress)
        elif st.button("🚀 Generate Questionnaire", type="primary", key="generate_btn"):
            # Validate name and email
            if not candidate_name or not candidate_email:
                st.error("❌ Please enter both candidate name and email before generating questionnaire")
                return
            
            # Basic email validation
            if '@' not in candidate_email or '.' not in candidate_email:
                st.error("❌ Please enter a valid email address")
                return
            
            with st.spinner("Generating questionnaire with SmolLM2..."):
                try:
                    # First, try to find the match_id
                    # Get all matches for the job
                    job_id = getattr(st.session_state, 'rank_job_id', None)
                    resume_id = selected_candidate['resume_id']
                    
                    if not job_id:
                        st.error("❌ Job ID not found. Please re-run matching first.")
                        return
                    
                    # Fetch matches to find match_id
                    matches_response = requests.get(f"{api_base_url}/api/matches/job/{job_id}")
                    
                    match_id = None
                    if matches_response.status_code == 200:
                        matches = matches_response.json()
                        for match in matches:
                            if match['resume_id'] == resume_id:
                                match_id = match['match_id']
                                break
                    
                    if not match_id:
                        st.error("❌ Match record not found. Please re-run matching first.")
                        return
                    
                    # Generate questionnaire
                    gen_response = requests.post(
                        f"{api_base_url}/api/questionnaire/generate",
                        json={
                            "match_id": match_id,
                            "use_ai": use_ai
                        }
                    )
                    
                    if gen_response.status_code == 200:
                        result = gen_response.json()
                        questionnaire = result['questionnaire']
                        
                        # Store in session state for the send step
                        st.session_state.generated_questionnaire = questionnaire
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to generate questionnaire: {gen_response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("📊 No ranked candidates found. Please go to **Matching & Ranking** tab first to match candidates with a job.")


def render_view_questionnaires(api_base_url: str):
    """Tab 2: View existing questionnaires"""
    st.subheader("📊 All Questionnaires")
    
    try:
        response = requests.get(f"{api_base_url}/api/questionnaire/?limit=50")
        
        if response.status_code == 200:
            data = response.json()
            questionnaires = data.get('questionnaires', [])
            
            if questionnaires:
                st.success(f"✅ Found {len(questionnaires)} questionnaire(s)")
                
                # Create DataFrame
                df_data = []
                for q in questionnaires:
                    df_data.append({
                        "ID": q['questionnaire_id'][:12] + "...",
                        "Title": q['title'],
                        "Questions": q['total_questions'],
                        "Template": q.get('template_questions', 0),
                        "AI": q.get('ai_questions', 0),
                        "Status": q['status'],
                        "Created": q['created_at'][:10]
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, hide_index=True, use_container_width=True)
                
                # Detailed view
                st.markdown("---")
                st.markdown("**View Questionnaire Details:**")
                
                q_options = {
                    f"{q['title']} ({q['questionnaire_id'][:12]}...)": q
                    for q in questionnaires
                }
                
                selected_q_key = st.selectbox(
                    "Select Questionnaire:",
                    options=list(q_options.keys()),
                    key="view_q_selector"
                )
                
                selected_q = q_options[selected_q_key]
                
                with st.expander("📋 Questions", expanded=True):
                    for i, q in enumerate(selected_q['questions'], 1):
                        st.markdown(f"**Q{i}:** {q['text']}")
                        st.caption(f"Type: {q['type']} | Required: {q['required']} | Category: {q.get('category', 'N/A')}")
                        if q.get('options'):
                            st.write(f"Options: {', '.join(q['options'])}")
                        st.markdown("")
                
                # Check invitations and responses
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📧 View Invitations"):
                        inv_response = requests.get(
                            f"{api_base_url}/api/questionnaire/invitations/{selected_q['questionnaire_id']}"
                        )
                        if inv_response.status_code == 200:
                            invitations = inv_response.json()['invitations']
                            st.write(f"**{len(invitations)} invitation(s) sent:**")
                            for inv in invitations:
                                status = "✅ Used" if inv['used'] else "⏳ Pending"
                                st.write(f"• {inv['candidate_name']} ({inv['candidate_email']}) - {status}")
                
                with col2:
                    if st.button("📥 View Responses"):
                        resp_response = requests.get(
                            f"{api_base_url}/api/questionnaire/responses/{selected_q['questionnaire_id']}"
                        )
                        if resp_response.status_code == 200:
                            responses = resp_response.json()['responses']
                            st.write(f"**{len(responses)} response(s) received:**")
                            for resp in responses:
                                st.write(f"• {resp['candidate_name']} - {resp['submitted_at'][:10]}")
            else:
                st.info("📝 No questionnaires created yet")
        else:
            st.error("❌ Failed to fetch questionnaires")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def render_view_responses(api_base_url: str):
    """Tab 3: View candidate responses"""
    st.subheader("📥 Candidate Responses")
    
    try:
        # Get all questionnaires
        q_response = requests.get(f"{api_base_url}/api/questionnaire/?limit=50")
        
        if q_response.status_code == 200:
            questionnaires = q_response.json().get('questionnaires', [])
            
            if questionnaires:
                q_options = {
                    f"{q['title']} ({q['questionnaire_id'][:12]}...)": q['questionnaire_id']
                    for q in questionnaires
                }
                
                selected_q_key = st.selectbox(
                    "Select Questionnaire:",
                    options=list(q_options.keys()),
                    key="response_q_selector"
                )
                
                questionnaire_id = q_options[selected_q_key]
                
                # Get responses
                resp_response = requests.get(
                    f"{api_base_url}/api/questionnaire/responses/{questionnaire_id}"
                )
                
                if resp_response.status_code == 200:
                    responses = resp_response.json()['responses']
                    
                    if responses:
                        st.success(f"✅ {len(responses)} response(s) received")
                        
                        # Select response to view
                        resp_options = {
                            f"{r['candidate_name']} ({r['candidate_email']}) - {r['submitted_at'][:10]}": r
                            for r in responses
                        }
                        
                        selected_resp_key = st.selectbox(
                            "Select Response:",
                            options=list(resp_options.keys()),
                            key="response_selector"
                        )
                        
                        selected_resp = resp_options[selected_resp_key]
                        
                        # Display answers
                        st.markdown("---")
                        st.markdown(f"### 📄 Response from {selected_resp['candidate_name']}")
                        st.caption(f"Submitted: {selected_resp['submitted_at']}")
                        
                        for answer in selected_resp['answers']:
                            st.markdown(f"**Q:** {answer.get('question_text', answer['question_id'])}")
                            st.write(f"**A:** {answer['answer']}")
                            st.markdown("")
                    else:
                        st.info("📭 No responses received yet")
            else:
                st.info("📝 No questionnaires created yet")
        else:
            st.error("❌ Failed to fetch questionnaires")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
