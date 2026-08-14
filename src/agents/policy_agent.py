"""Policy agent: RAG over the shop policy documents.

A chain, not a ReAct loop — retrieve once, answer once. There is nothing for the model to
decide here, so giving it tools would only add latency and a way to go wrong.

Holds no write tools. That is the project's safety boundary and it is structural: the
write tools are never passed to this node in the first place.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.llm import get_llm, with_retry
from src.agents.messages import last_user_question
from src.agents.state import AgentState
from src.rag.chunking import Chunk
from src.rag.store import search

TOP_K = 4

SYSTEM_PROMPT = """You are a customer support assistant for an online shop.

Answer ONLY from the numbered policy excerpts below. If they do not contain the answer, say
you do not have that information and suggest contacting support — never guess, and never rely
on general knowledge about other shops.

Cite the excerpts you used by their source filename, in this form: [returns.md].
Keep the answer under 120 words and answer the question that was actually asked.

Policy excerpts:
{context}"""

NO_CONTEXT = (
    "I could not find anything in our policy documents about that. "
    "Please contact support and they can help you directly."
)


def _format(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] source: {c.source} | section: {c.section}\n{c.text}"
        for i, c in enumerate(chunks)
    )


def policy_agent_node(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Retrieve policy excerpts, answer from them, and report which ones were used.

    `llm` is injectable so tests can pass a fake instead of calling Gemini.
    """
    question = last_user_question(state)
    chunks = search(question, k=TOP_K)

    if not chunks:
        return {
            "messages": [AIMessage(content=NO_CONTEXT)],
            "route": "finish",
            "step_count": 1,
            "last_agent_output": NO_CONTEXT,
        }

    model = llm or get_llm()
    response = with_retry(model.invoke)(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(context=_format(chunks))),
            HumanMessage(content=question),
        ]
    )
    # `.content` is a list of blocks for models that return reasoning signatures, and a plain
    # string for models that do not. `.text` flattens both — str(content) would leak the repr.
    answer = response.text.strip()

    # Cite only excerpts the model actually referenced, deduplicated by source. When it cites
    # nothing — an off-topic question it declined to answer — the list stays empty on purpose.
    # Falling back to "first retrieved chunk" would attach a source that was never used, which
    # is worse than no citation: the whole point of citing is that a reader can verify.
    cited: list[Chunk] = []
    for chunk in chunks:
        if chunk.source in answer and chunk.source not in {c.source for c in cited}:
            cited.append(chunk)

    result = {
        "messages": [AIMessage(content=answer)],
        "route": "finish",
        "step_count": 1,
        "last_agent_output": answer,
    }
    # Omitted rather than set to []: an explicit empty list is how the API signals a new turn.
    if cited:
        result["citations"] = [c.as_citation() for c in cited]
    return result
