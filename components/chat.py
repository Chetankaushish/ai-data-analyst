import streamlit as st
import pandas as pd
from utils.chat_memory import ChatMemory

memory = ChatMemory()
def render_chat(df, ask_ai):

    st.markdown("## 🤖 AI Data Chat")

    st.caption("Ask anything about your uploaded dataset.")

    # ---------------- Chat Memory ---------------- #

    suggested_questions = [
        "Give me executive summary",
        "Top 10 rows",
        "Find missing values",
        "Describe this dataset",
        "Show important insights",
        "Which numeric columns exist?"
    ]

    st.markdown("### 💡 Suggested Questions")

    cols = st.columns(3)

    for i, question in enumerate(suggested_questions):

        if cols[i % 3].button(question):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.spinner("Thinking..."):

                answer = ask_ai(df, question)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

    st.divider()

    # ---------------- Show Chat ---------------- #

    memory.render()

    # ---------------- Chat Input ---------------- #

    prompt = st.chat_input(
        "Ask anything about your data..."
    )

    if prompt:
        
        memory.add_user(prompt)

        with st.chat_message("user"):

            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing dataset..."):

                answer = ask_ai(df, prompt)

                st.markdown(answer)

        memory.add_ai(answer)

    st.divider()

    # ---------------- Dataset Info ---------------- #

    with st.expander("📄 Dataset Information"):

        c1, c2 = st.columns(2)

        c1.metric("Rows", df.shape[0])

        c2.metric("Columns", df.shape[1])

        st.write("### Column Types")

        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Datatype": df.dtypes.astype(str)
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True
        )