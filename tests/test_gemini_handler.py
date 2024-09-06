import pytest
from unittest.mock import patch, MagicMock
from sgpt.handlers.gemini_handler import GeminiHandler, GeminiChatHandler
from sgpt.role import SystemRole, DefaultRoles

@pytest.fixture
def mock_genai():
    with patch('sgpt.handlers.gemini_handler.genai') as mock:
        yield mock

@pytest.fixture
def gemini_handler(mock_genai):
    role = SystemRole.get(DefaultRoles.DEFAULT.value)
    return GeminiHandler(role, md=True)

@pytest.fixture
def gemini_chat_handler(mock_genai):
    role = SystemRole.get(DefaultRoles.DEFAULT.value)

    return GeminiChatHandler(True, role, md=True)

def test_gemini_handler_initialization(gemini_handler, mock_genai):
    assert isinstance(gemini_handler, GeminiHandler)
    mock_genai.configure.assert_called_once()
    mock_genai.GenerativeModel.assert_called_once_with('gemini-1.5-flash')

def test_gemini_handler_make_messages(gemini_handler):
    prompt = "Test prompt"
    messages = gemini_handler.make_messages(prompt)
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == prompt

def test_gemini_handler_get_completion(gemini_handler, mock_genai):
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_genai.GenerativeModel().generate_content.return_value = [mock_response]

    completion = list(gemini_handler.get_completion(
        model="gemini-1.5-flash",
        temperature=0.7,
        top_p=1.0,
        messages=[{"role": "user", "content": "Test prompt"}],
        functions=None,
        caching=False
    ))

    assert completion == ["Test response"]
    mock_genai.GenerativeModel().generate_content.assert_called_once()

def test_gemini_chat_handler_initialization(gemini_chat_handler, mock_genai):
    assert isinstance(gemini_chat_handler, GeminiChatHandler)
    mock_genai.configure.assert_called_once()
    mock_genai.GenerativeModel.assert_called_once_with('gemini-1.5-flash')

def test_gemini_chat_handler_get_completion(gemini_chat_handler, mock_genai):
    mock_response = MagicMock()
    mock_response.text = "Test chat response"
    mock_genai.GenerativeModel().start_chat().send_message.return_value = [mock_response]

    completion = list(gemini_chat_handler.get_completion(
        model="gemini-1.5-flash",
        temperature=0.7,
        top_p=1.0,
        prompt="Test chat prompt",
        functions=None,
        caching=False
    ))

    assert completion == ["Test chat response"]
    mock_genai.GenerativeModel().start_chat().send_message.assert_called_once()