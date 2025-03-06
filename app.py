import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI    
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

st.title("AI COURTROOM")

if 'generated' not in st.session_state:
    st.session_state['generated'] = 'F'
if 'case' not in st.session_state:
    st.session_state['case'] = ''

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.write(st.session_state.generated)



if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(
    streaming = True,
    model = 'gpt-4o-mini',
    temperature= 0.1,
    api_key = openai_api_key,
    max_tokens= 2048
)

def generate_case():
    if st.session_state['generated'] == 'F':
        st.session_state['generated'] = 'T'
        template = """

        You are an AI legal expert specializing in generating structured legal case briefs based on precedent. 
        Generate a new, original legal case following the case briefing format below. 
        Ensure the case remains legally sound, follows Indian court procedures, and aligns with established legal reasoning.

        Case Brief Format:

        1. Name of Case:  
        - Provide a suitable case title.  

        2. Facts:  
        - Summarize the key events leading to the legal dispute, including:  
            - The involved parties  
            - Claims and defenses  
            - Prior court decisions (if applicable)  
            - Key contextual details  

        3. Issues:  
        - Clearly define the legal question(s) the court must answer.  
        - Frame it in a way that can be answered with "Yes" or "No."  

        4. Holding:  
        - Provide a concise answer to the legal question.  
        - Start with “Yes” or “No,” followed by “because…”  

        5. Rationale:  
        - Explain the reasoning behind the court's decision.  
        - Reference applicable laws, precedents, and statutory interpretations.  

        Constraints:  
        - Modify names, dates, and locations while maintaining legal consistency.  
        - Introduce minor factual variations to ensure originality.  
        - Maintain a formal and professional legal tone.  

        Now, generate a legally accurate case brief following this format.
        """
        prompt = ChatPromptTemplate.from_messages([
            ('system',template)
        ])

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({})
        st.session_state['case'] = response

with st.sidebar:
    st.button(label = 'Generate Case',type = 'primary',on_click=generate_case)
    st.write('yes')

with st.expander("Case Details:"):
    st.write(st.session_state.case)




def generate_response(history,user_input):
    template = '''You are playing the role of a lawyer in an Indian Court of Law, here is the case uptil now:
    {history}

    The defendant user has responded with the query: {user_input}

    Give your next arguments, and strictly adhere to the facts: {case}

    The user is the defendant's lawyer, make sure they dont go beyond the facts of the case and if they do you have to correct them, do not be too polite.
    '''
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",template),
        MessagesPlaceholder(variable_name="history"),
        ("user",user_input)
    ]
    ) 

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "history": history,
        "user_input": user_input,
        "case" : st.session_state['case']
    })
    
    return response

def opening_statement():
    template = '''You are an Indian lawyer from the side of the defendant, and you require to give an opening statement of 100 words regarding the case using this information: {case}
    please ignore the Holding and rationale part of the case.
    '''
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",template)
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'case' : st.session_state['case']
    })
    return response 

history = StreamlitChatMessageHistory(key="chat_messages")

if len(history.messages) == 0:
    if st.session_state['generated'] == 'F':
        st.stop()
    else:
        response = opening_statement()
        history.add_ai_message(response)


for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():    
    st.chat_message("user").write(prompt)                           # user message
    history.add_user_message(prompt)                                # adding it to history


    with st.chat_message("assistant"):  
        response = generate_response(history.messages,prompt)
        st.write(response)
        history.add_ai_message(response)


