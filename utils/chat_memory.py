import streamlit as st


class ChatMemory:

    def __init__(self):

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    # ----------------------------

    def add_user(self, message):

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": message
            }
        )

    # ----------------------------

    def add_ai(self, message):

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": message
            }
        )

    # ----------------------------

    def get_messages(self):

        return st.session_state.chat_history

    # ----------------------------

    def clear(self):

        st.session_state.chat_history = []

    # ----------------------------

    def render(self):

        for message in self.get_messages():

            with st.chat_message(message["role"]):

                st.markdown(
                    message["content"]
                )

    # ----------------------------

    def context(self, limit=10):

        history = self.get_messages()

        if len(history) > limit:

            history = history[-limit:]

        prompt = ""

        for msg in history:

            prompt += (
                f"{msg['role']}: "
                f"{msg['content']}\n"
            )

        return prompt

    # ----------------------------

    def size(self):

        return len(
            st.session_state.chat_history
        )