
import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("lm_eval")

from veeksha.config.generator.session import LmevalSessionGeneratorConfig
from veeksha.generator.session.lmeval import LMEvalSessionGenerator
from veeksha.core.seeding import SeedManager
from veeksha.types import LMEvalOutputType, ChannelModality

@pytest.fixture
def mock_tokenizer_provider():
    provider = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda x: [1] * len(x.split())
    tokenizer.decode.side_effect = lambda x: x
    provider.for_modality.return_value = tokenizer
    return provider

@pytest.fixture
def mock_seed_manager():
    return SeedManager(seed=42)

@pytest.fixture
def lmeval_config():
    return LmevalSessionGeneratorConfig(
        tasks=["test_task"],
        num_fewshot=0
    )

@pytest.fixture
def mock_lm_eval_components():
    with patch("veeksha.generator.session.lmeval.TaskManager") as mock_tm, \
         patch("veeksha.generator.session.lmeval.get_task_dict") as mock_gtd, \
         patch("veeksha.generator.session.lmeval.get_task_list") as mock_gtl:
        
        # Setup Task
        mock_task = MagicMock()
        mock_task.instances = []
        
        # Create some instances
        instance1 = MagicMock()
        instance1.doc_id = 0
        instance1.idx = 0
        instance1.request_type = str(LMEvalOutputType.GENERATE_UNTIL)
        instance1.args = ("context", {"max_gen_toks": 10})
        instance1.repeats = 1
        
        instance2 = MagicMock()
        instance2.doc_id = 1
        instance2.idx = 1
        instance2.request_type = str(LMEvalOutputType.LOGLIKELIHOOD)
        instance2.args = ("context", "target")
        instance2.repeats = 1
        
        mock_task.instances = [instance1, instance2]
        
        # Mock doc_iterator to return doc_ids
        mock_task.doc_iterator.return_value = [(0, None), (1, None)]
        
        # Setup TaskOutput (wrapper around task)
        mock_task_output = MagicMock()
        mock_task_output.task = mock_task
        mock_task_output.task_name = "test_task"
        
        mock_gtd.return_value = {"test_task": mock_task}
        mock_gtl.return_value = [mock_task_output]
        
        yield mock_tm, mock_gtd, mock_gtl, mock_task

def test_lmeval_generator_initialization(lmeval_config, mock_seed_manager, mock_tokenizer_provider, mock_lm_eval_components):
    generator = LMEvalSessionGenerator(lmeval_config, mock_seed_manager, mock_tokenizer_provider)
    assert generator.capacity() == 2  # 2 docs

def test_lmeval_generator_generate_sessions(lmeval_config, mock_seed_manager, mock_tokenizer_provider, mock_lm_eval_components):
    generator = LMEvalSessionGenerator(lmeval_config, mock_seed_manager, mock_tokenizer_provider)
    
    # Session 1 (Generate Until)
    session1 = generator.generate_session()
    assert len(session1.requests) == 1
    req1 = session1.requests[0]
    assert req1.requested_output is not None
    assert req1.requested_output.text is not None
    assert req1.requested_output.text.target_tokens == 10
    assert req1.metadata["lmeval_request_type"] == str(LMEvalOutputType.GENERATE_UNTIL)
    
    # Session 2 (LogLikelihood)
    session2 = generator.generate_session()
    assert len(session2.requests) == 1
    req2 = session2.requests[0]
    # Loglikelihood defaults to target_tokens=1 in the implementation
    assert req2.requested_output is not None
    assert req2.requested_output.text is not None
    assert req2.requested_output.text.target_tokens == 1
    assert req2.metadata["lmeval_request_type"] == str(LMEvalOutputType.LOGLIKELIHOOD)
    
    # Exhaustion
    with pytest.raises(StopIteration):
        generator.generate_session()

def test_lmeval_generator_max_sessions(lmeval_config, mock_seed_manager, mock_tokenizer_provider, mock_lm_eval_components):
    # Limit to 1 session
    generator = LMEvalSessionGenerator(lmeval_config, mock_seed_manager, mock_tokenizer_provider, max_sessions=1)
    assert generator.capacity() == 1
    
    generator.generate_session()
    with pytest.raises(StopIteration):
        generator.generate_session()
