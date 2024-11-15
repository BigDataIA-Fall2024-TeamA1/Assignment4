# streamlit_app.py

import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000/publications"

def main():
    st.title("Publication Query Application")

    # Fetch publications from the FastAPI service
    try:
        response = requests.get(API_BASE_URL)
        if response.status_code == 200:
            publications = response.json().get("publications", [])
        else:
            st.error(f"Failed to retrieve publications: {response.status_code} - {response.text}")
            return
    except Exception as e:
        st.error(f"An error occurred while fetching publications: {str(e)}")
        return

    if not publications:
        st.warning("No publications found.")
        return

    # Create a dropdown list to select a publication
    selected_publication = st.selectbox("Select a publication", publications)

    st.write(f"You selected: {selected_publication}")

    query = st.text_input("Enter your query")

    # Choose an agent (tool)
    agent_options = ["rag_search_filter", "web_search"]
    selected_agent = st.selectbox("Select an agent", agent_options)

    if st.button("Submit Query"):
        # Build request data
        data = {
            "input": query,
            "publication": selected_publication if selected_agent == "rag_search_filter" else None
        }
        try:
            # Send request to FastAPI service
            response = requests.post("http://localhost:8000/query", json=data)
            if response.status_code == 200:
                report = response.json()
                st.subheader("Introduction")
                st.write(report["introduction"])
                st.subheader("Research Steps")
                st.write(report["research_steps"])
                st.subheader("Main Body")
                st.write(report["main_body"])
                st.subheader("Conclusion")
                st.write(report["conclusion"])
                st.subheader("Sources")
                st.write(report["sources"])
            else:
                st.error(f"Request failed: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()