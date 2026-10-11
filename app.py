
import streamlit as st
import pandas as pd

from nlp_engine import analyze_error

st.set_page_config(
    page_title="AI Error Log Explainer",
    page_icon="🔍",
    layout="wide"
)

# Store history for the current Streamlit session
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔍 AI-Powered Error Log Explainer")

st.write(
    "Understand Python and Java errors using NLP, "
    "text classification, and similarity matching."
)

st.info("Paste an error message or complete traceback below.")

error_log = st.text_area(
    "Error Log",
    height=220,
    placeholder=(
        'File "app.py", line 24\n'
        "NameError: name 'total' is not defined"
    )
)

if st.button("Analyze Error", type="primary"):

    if not error_log.strip():
        st.warning("Please enter an error log.")

    else:
        with st.spinner("Analyzing error log..."):
            result = analyze_error(error_log)

        if "error" in result:
            st.error(result["error"])

        else:
            # Save each analysis in session history
            st.session_state.analysis_history.insert(
                0,
                {
                    "Input": error_log,
                    "Error Type": result["predicted_type"],
                    "Language": result["language"],
                    "Similarity (%)": result["similarity"],
                    "Cause": result["cause"],
                    "Solution": result["solution"]
                }
            )

            st.success("Analysis complete!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Predicted Error")
                st.write(result["predicted_type"])

            with col2:
                st.subheader("Language")
                st.write(result["language"])

            with col3:
                st.subheader("Text Similarity")
                st.write(f"{result['similarity']}%")
                st.progress(
                    min(result["similarity"] / 100, 1.0)
                )

            st.divider()

            st.subheader("🔎 Extracted Details")

            detail1, detail2, detail3 = st.columns(3)

            with detail1:
                files = result.get("filenames", [])
                st.write("**Filename(s)**")
                st.write(", ".join(files) if files else "Not detected")

            with detail2:
                lines = result.get("line_numbers", [])
                st.write("**Line number(s)**")
                st.write(", ".join(lines) if lines else "Not detected")

            with detail3:
                exceptions = result.get("exceptions_found", [])
                st.write("**Exception(s)**")
                st.write(
                    ", ".join(exceptions)
                    if exceptions else "Not detected"
                )

            st.divider()

            st.subheader("💡 Likely Cause")
            st.write(result["cause"])

            st.subheader("🛠️ Suggested Solution")
            st.write(result["solution"])

            with st.expander("View matched dataset example"):
                st.code(result["matched_error"])

            st.caption(
                "Verify suggested fixes before applying them. "
                "Similarity is not a measure of diagnostic accuracy."
            )


# Analysis history
st.divider()
st.header("📚 Analysis History")

history = st.session_state.analysis_history

if history:
    st.write(f"Errors analyzed this session: {len(history)}")

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Analysis History (CSV)",
        data=csv_data,
        file_name="error_analysis_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

else:
    st.write("No errors analyzed in this session yet.")

st.divider()

st.subheader("How it works")

st.markdown(
    """
    1. **Preprocessing:** Cleans the error-log text.
    2. **Classification:** Uses TF-IDF features and a linear SVM.
    3. **Similarity matching:** Finds the closest dataset example.
    4. **Explanation:** Displays the likely cause and solution.
    5. **History:** Keeps session analyses and allows CSV export.
    """
)

st.caption(
    "NLP Mini Project | Python • scikit-learn • Streamlit"
)
