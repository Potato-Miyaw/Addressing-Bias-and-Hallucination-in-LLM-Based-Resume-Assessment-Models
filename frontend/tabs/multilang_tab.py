import streamlit as st
import requests

def render(api_base_url: str):
    st.header("🌍 Multi-Language Resume Support")
    st.markdown("Translate and verify resumes that were uploaded in the **Resumes** tab.")
    
    # Check if resumes exist in session state
    if 'resumes_data' not in st.session_state or not st.session_state.resumes_data:
        st.warning("No resumes found. Please upload resumes in the 'Resumes' tab first.")
        return

    # List available resumes
    resume_options = {r['filename']: r for r in st.session_state.resumes_data}
    selected_filename = st.selectbox("Select Resume to Process", list(resume_options.keys()))
    
    if selected_filename:
        selected_resume = resume_options[selected_filename]
        original_data = selected_resume.get('data', {})
        
        # Get text from the stored resume data
        # Note: 'text' field is populated by the upload endpoint
        original_text = original_data.get('text', '')
        
        if not original_text:
            st.error("No text content found in this resume.")
        else:
            st.markdown("### Original Resume Content (Snippet)")
            st.info(original_text[:500] + "...")
            
            if st.button("Translate & Process"):
                with st.spinner("Translating to English and extracting info..."):
                    try:
                        # Call the new text processing endpoint
                        response = requests.post(
                            f"{api_base_url}/api/multilang/process-text",
                            json={"text": original_text}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.success("Translation & Processing Complete!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original Text")
                                st.text_area("Source", data["original_text_preview"], height=200, key="orig_preview")
                                
                            with col2:
                                st.subheader("Translated English")
                                st.text_area("English", data["english_text_preview"], height=200, key="eng_preview")
                            
                            st.markdown("---")
                            st.subheader("📊 Extracted Information (From Translated Text)")
                            
                            parsed = data["parsed_data"]
                            
                            # Display metrics
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Name", parsed.get('name', 'Unknown'))
                            c2.metric("Email", parsed.get('email_address', 'N/A'))
                            phone = parsed.get('contact_number', 'N/A')
                            if isinstance(phone, list):
                                phone = ", ".join(phone) if phone else "N/A"
                            c3.metric("Phone", phone)
                            
                            st.markdown("**Skills:**")
                            skills = parsed.get('skills', [])
                            if skills:
                                st.markdown(" ".join([f"`{s}`" for s in skills]))
                            else:
                                st.write("No skills detected.")
                                
                        else:
                            st.error(f"Error: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")
