"""The agent loop, written with LangGraph.

read -> route -> correct -> verify -> (retry once | escalate | accept)

Nothing here is magic. The value is that every hop is a named node, so the
trace in Langfuse shows exactly which path a request took, how many LLM
calls it cost, and why a line ended up in the review queue. That is the
AgentOps part of the project.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from nastaliqops.config import load_params
from nastaliqops.llm.corrector import LlmCorrector


class PipelineState(TypedDict, total=False):
    image: object
    text: str
    confidence: float
    variant: str
    route: str
    attempts: int
    llm_calls: int
    changed: bool
    needs_review: bool
    reason: str


def build_graph(predictor):
    params = load_params()
    router_params = params["router"]

    corrector = None
    try:
        corrector = LlmCorrector()
    except Exception as error:
        print("corrector disabled: " + str(error))

    def node_read(state):
        result = predictor.predict_line(state["image"])
        state["text"] = result["text"]
        state["confidence"] = result["confidence"]
        state["variant"] = result["variant"]
        state["attempts"] = state.get("attempts", 0)
        state["llm_calls"] = state.get("llm_calls", 0)
        return state

    def node_route(state):
        confidence = state["confidence"]
        if confidence < router_params["review_threshold"]:
            state["route"] = "escalate"
            state["reason"] = "confidence below the review threshold"
        elif confidence < router_params["confidence_threshold"]:
            state["route"] = "correct"
            state["reason"] = "confidence below the trust threshold, asking the llm"
        else:
            state["route"] = "accept"
            state["reason"] = "model is confident enough on its own"
        return state

    def node_correct(state):
        if corrector is None:
            state["route"] = "escalate"
            state["reason"] = "no llm configured"
            return state

        attempts = state.get("attempts", 0)
        corrected = corrector.correct(state["text"], state["confidence"])

        state["text"] = corrected["text"]
        state["changed"] = corrected["changed"]
        state["attempts"] = attempts + 1
        state["llm_calls"] = state.get("llm_calls", 0) + 1

        if corrected["needs_review"]:
            state["route"] = "escalate"
            state["reason"] = "the llm refused to repair this line"
        else:
            state["route"] = "verify"
        return state

    def node_verify(state):
        """Cheap self check. An empty or absurdly short repair is a failure."""
        text = state.get("text", "")
        if len(text.strip()) < 2:
            if state.get("attempts", 0) < 2:
                state["route"] = "correct"
                state["reason"] = "repair came back empty, retrying once"
            else:
                state["route"] = "escalate"
                state["reason"] = "repair failed twice"
        else:
            state["route"] = "accept"
        return state

    def node_accept(state):
        state["needs_review"] = False
        return state

    def node_escalate(state):
        state["needs_review"] = True
        return state

    def choose_after_route(state):
        return state["route"]

    def choose_after_correct(state):
        return state["route"]

    def choose_after_verify(state):
        return state["route"]

    graph = StateGraph(PipelineState)
    graph.add_node("read", node_read)
    graph.add_node("route", node_route)
    graph.add_node("correct", node_correct)
    graph.add_node("verify", node_verify)
    graph.add_node("accept", node_accept)
    graph.add_node("escalate", node_escalate)

    graph.set_entry_point("read")
    graph.add_edge("read", "route")

    graph.add_conditional_edges(
        "route",
        choose_after_route,
        {"accept": "accept", "correct": "correct", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "correct",
        choose_after_correct,
        {"verify": "verify", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "verify",
        choose_after_verify,
        {"accept": "accept", "correct": "correct", "escalate": "escalate"},
    )

    graph.add_edge("accept", END)
    graph.add_edge("escalate", END)

    return graph.compile()
