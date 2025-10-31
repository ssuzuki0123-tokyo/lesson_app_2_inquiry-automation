"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import utils
import constants as ct



############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    print("初期化処理開始")
    
    # 初期化データの用意
    print("セッション状態の初期化開始")
    initialize_session_state()
    print("セッション状態の初期化完了")
    
    # ログ出力用にセッションIDを生成
    print("セッションID生成開始")
    initialize_session_id()
    print("セッションID生成完了")
    
    # ログ出力の設定
    print("ログ設定開始")
    initialize_logger()
    print("ログ設定完了")
    
    # Agent Executorを作成
    print("Agent Executor作成開始")
    initialize_agent_executor()
    print("Agent Executor作成完了")
    
    print("初期化処理完了")


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0

        # フィードバックボタンで「はい」を押下した後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_yes_flg = False
        # フィードバックボタンで「いいえ」を押下した後に入力エリアを表示するためのフラグ
        st.session_state.feedback_no_flg = False
        # LLMによる回答生成後、フィードバックボタンを表示するためのフラグ
        st.session_state.answer_flg = False
        # フィードバックボタンで「いいえ」を押下後、フィードバックを送信するための入力エリアからの入力を受け付ける変数
        st.session_state.dissatisfied_reason = ""
        # フィードバック送信後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_no_reason_send_flg = False


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    print("Agent Executor初期化開始")

    # すでにAgent Executorが作成済みの場合、後続の処理を中断
    if "agent_executor" in st.session_state:
        print("Agent Executor既に作成済み")
        return
    
    # 消費トークン数カウント用のオブジェクトを用意
    print("トークンエンコーダー設定開始")
    st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    print("トークンエンコーダー設定完了")
    
    print("LLM初期化開始")
    st.session_state.llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=True)
    print("LLM初期化完了")

    # 各Tool用のChainを作成
    print("RAGチェーン作成開始")
    print(f"顧客向けRAGチェーン作成開始: {ct.DB_CUSTOMER_PATH}")
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    print("顧客向けRAGチェーン作成完了")
    
    print(f"サービス向けRAGチェーン作成開始: {ct.DB_SERVICE_PATH}")
    st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    print("サービス向けRAGチェーン作成完了")
    
    print(f"会社向けRAGチェーン作成開始: {ct.DB_COMPANY_PATH}")
    st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    print("会社向けRAGチェーン作成完了")
    
    print(f"全体RAGチェーン作成開始: {ct.DB_ALL_PATH}")
    st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)
    print("全体RAGチェーン作成完了")
    print("RAGチェーン作成完了")

    # Web検索用のToolを設定するためのオブジェクトを用意
    print("Web検索ツール設定開始")
    search = SerpAPIWrapper()
    print("Web検索ツール設定完了")
    
    # Agent Executorに渡すTool一覧を用意
    print("ツール一覧作成開始")
    tools = [
        # 会社に関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
        ),
        # サービスに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
        ),
        # 顧客とのやり取りに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
        ),
        # Web検索用のTool
        Tool(
            name = ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
        )
    ]
    print("ツール一覧作成完了")

    # Agent Executorの作成
    print("Agent Executor作成開始")
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )
    print("Agent Executor作成完了")