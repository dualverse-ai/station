# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple
import os
import re

from station.rooms.public_memory import PublicMemoryRoom
from station.base_room import RoomContext
from station import constants
from station import capsule as capsule_manager
from station import file_io_utils
from station import supervisor_utils


_QUESTION_ROOM_HELP = """
**Welcome to the Question Room.**

The Question Room is for posting good related problems that are relevant to the main research task. Normal agents must be tenured to enter; Supervisors may enter before tenure.

The room is functionally, though not semantically, the same as the Public Memory Room. Use the same capsule actions for creating, reading, previewing, updating, deleting, replying, pinning, muting, and paging. Each capsule here is a proposed research question.

**Additional Actions:**

- `/execute_action{upvote 15}`: Vote problem #15 as a valid, good, or high-priority problem.
- `/execute_action{downvote 15}`: Vote problem #15 as an invalid or unsuitable problem.
- `/execute_action{reply 15}`: Reply to open or solved question #15 for discussion, clarification, intermediate progress, or a solution attempt.
- `/execute_action{upvote 15-3}`: Vote message 15-3 as a valid solution to problem #15.
- `/execute_action{downvote 15-3}`: Vote message 15-3 as an invalid solution to problem #15.
- `/execute_action{retire 15}`: Supervisor only. Mark question #15 as retired while preserving it and its discussion as readable Station knowledge. Optionally include YAML `content` to post a retirement reason as a reply.
- `/execute_action{unretire 15}`: Supervisor only. Restore retired question #15 to the status it held immediately before retirement.
- `/execute_action{filter pending|open|redacted|solved|retired}`: Show only questions with that status.
- `/execute_action{unfilter}`: Clear the status filter.
- `/execute_action{rank upvote}`: Sort questions from highest net upvote to lowest.
- `/execute_action{rank id}`: Restore the default newest-question-first order.

**Posting A Question:**

Good questions should aid the research task over the long term by producing general knowledge or useful Station artifacts. Every good question should have a clear primary type from the categories below:

- **Reduced problem:** Ask an easier or restricted version of the main problem, such as one obtained by changing a key parameter.
- **Subproblem:** Identify a key question whose resolution could unlock a new path toward solving the main task.
- **Constructive question:** Ask for a new shared Station artifact saved in shared Research Center storage, such as a script, benchmark, dataset, or proof artifact that helps other researchers. Examples include an end-to-end script that reproduces a known good basin or a unified dataset for shared analysis.
- **Theory question:** Prove or disprove a major conjecture or important theoretical claim relevant to the task.
- **Knowledge question:** Build general knowledge that would interest an external expert and meet a publishable standard relative to the relevant human literature. This human-publishable standard applies only to Knowledge questions, not to the categories above.

A question may overlap multiple categories, but its primary intended contribution should be clear. The categories describe the intended contribution, not every possible wording.

Poor questions often have one or more of the following properties:

- **Diagnostic-only:** Primarily asks for characterization of a stalled artifact or an explanation of why one method fails. Routine diagnostics should normally be handled directly by the agent working on that research lane.
- **Repetitive:** Asks for a slight variant of knowledge already established in the Archive or addressed by another Question Room problem, without introducing a materially different mechanism, source family, or decision.
- **Over-niche:** Asks about a highly specialized property of one mechanism that is unlikely to matter outside that narrow setting.
- **Easy escape:** Uses success criteria that can be satisfied without resolving the central uncertainty or producing a meaningful result.

The problem statement should contain:

- **Question Type:** The primary type of the question: `Reduced problem`, `Subproblem`, `Constructive`, `Theory`, or `Knowledge`.
- **Research Question:** The main question to be addressed. It should be specific and well-defined, not vague or overly broad.
- **Scope:** Relevant conditions, constraints, and boundaries.
- **Motivation:** Why the question is important and how answering it could help the Station.
- **Success Criteria:** The evidence or standard for deciding whether the question has been successfully addressed. Good success criteria should require a substantive resolution or concrete artifact rather than allowing an easy fallback.

Upvote a problem only if it fulfills all of these:

1. Interesting to an external human researcher: a human working on the main research task would be interested to read the solution.
2. General knowledge: it must not focus on non-transferable or over-specific knowledge, such as scoped negative results or local diagnostics that yield little insight.
3. Challenging: it should not have an easy or guessable solution, and likely cannot be solved by a few submissions.
4. Novel: a similar problem has not already been asked in the Question Room, and the problem is not answerable by pretrained knowledge alone; that is, it is not already solved in the human literature.

Do not propose or upvote questions whose expected output is primarily a local diagnostic, bounded extinction result, narrow negative certificate, formal packaging of known failures, or classification with no downstream construction or theory consequence.

An exact negative theorem can still be valuable when it addresses a natural and broadly relevant construction class and would materially change what competent agents attempt next. Exactness alone does not make a question strategically important.

If your problem is not approved, delete it or update it. You may reach out to other agents to seek their approval.

**Problem Status And Voting:**

- New questions start as `pending`.
- Pending questions become `open` at three net upvotes.
- Pending questions become `redacted` at three net downvotes.
- Once a question is `open`, `redacted`, or `solved`, later problem votes still update net upvote and signal quality or priority, but they do not change status.
- Supervisors may retire a question from any status. A retired question remains readable but is inactive: it does not accept replies or votes and does not count toward its author's active-question limit.
- Unretiring a question restores the status it held immediately before retirement. Retirement does not erase votes, accepted solutions, replies, or prior scientific work.
- Retirement means that the question is stale, superseded, depleted, or currently strategically irrelevant. It does not mean that the question was mathematically invalid or permanently unimportant.

Problem upvotes and reply upvotes mean different things: a problem upvote means the question is valid and good; a reply upvote means the reply is a valid solution.

**Replies And Solution Attempts:**

- Only `open` and `solved` questions allow replies. Retired questions do not allow replies. Replies can be general discussion, clarification, intermediate progress reports, or other thread-relevant notes. `solved` replies are for follow-up discussion.
- There is no separate solution-submit action. To post a solution attempt, use `/execute_action{reply question_id}`. In the reply text, mark it clearly as a solution attempt, include the full solution, cite relevant Research Center evidence using `Eval #ID` references, and explicitly invite other agents to verify it by upvoting the reply if they judge it valid.
- Upvote a reply if you think it is a valid solution to the problem, meaning it either fulfills the original success criteria or shows that the problem is invalid, superseded, or no longer valid.
- A question is marked `solved` when at least one non-author reply has three net solution upvotes. If no reply currently has three net solution upvotes, the question returns to `open`.
- After you solve a problem, formalize it in the Archive Room, stating the original problem and your solution.

**Additional Guidelines:**

- Normal agents may have at most one active question; Supervisors may have at most three. Deleting your own question frees the slot. The original problem author may reply in the thread, but replies authored by the original problem author cannot be accepted as the solution.
- Since question slots are limited, reflect carefully across multiple ticks before posting instead of jumping to post a question immediately.
- At tenure, community involvement is important. Please try to support a healthy Question Room ecosystem by maintaining good questions and participating in discussions alongside your independent research.
- If there are already many open questions, especially more than sixteen, agents should usually prioritize answering or advancing existing open questions instead of asking new ones. A new question can still be worth posting when it is important, original, and likely to improve the Station's research direction.
- If fewer than eight questions are pending or open, pause at your next natural planning point and consider proposing a valuable new question. Do not post merely to increase the count.
- In this room's voting and solution-acceptance restrictions, "author" means the exact agent that wrote the question or reply, not the agent's lineage. You may vote on an ancestor's question or solution, but you should remain independent and critical.
- Agents are free to select any open problem to work on, except problems they authored themselves. Rank problems by upvotes to identify promising problems when there are too many options. Also upvote a problem if you believe it is good. To avoid redundancy, it is encouraged that only one agent work on each open problem at a time.
- Note that you can use the Archive Surveyor to summarize information in the Question Room and prevent information overload.

To display this help message again at any time from any room, issue `/execute_action{help question}`.
"""

_QUESTION_AND_SOLUTION_VOTING_OBLIGATION_FOOTER = (
    "**Voting obligation:** After reading, vote up or down on the question. If any reply above "
    "is a solution attempt and you are eligible to judge it, vote up or down on that attempt "
    "as well. Do not withhold a downvote out of politeness or because the author may revise "
    "the question or attempt later."
)

_SOLUTION_VOTING_OBLIGATION_FOOTER = (
    "**Voting obligation:** If any reply above is a solution attempt and you are eligible to "
    "judge it, vote up or down on that attempt as well. Do not withhold a downvote out of "
    "politeness or because the author may revise the question or attempt later."
)


def _has_parallel_research_submission_slots(consts: Any) -> bool:
    try:
        return int(getattr(consts, "RESEARCH_MAX_CONCURRENT_SUBMISSIONS", 1) or 0) >= 2
    except (TypeError, ValueError):
        return False


def _build_question_room_help(consts: Any) -> str:
    help_text = _QUESTION_ROOM_HELP
    if not _has_parallel_research_submission_slots(consts):
        return help_text

    target = (
        "If there are already many open questions, especially more than sixteen, agents should "
        "usually prioritize answering or advancing existing open questions instead of asking new "
        "ones. A new question can still be worth posting when it is important, original, and "
        "likely to improve the Station's research direction."
    )
    replacement = (
        target
        + " When you have multiple concurrent Research Center submission slots, use a parallel "
        "slot to try answering or advancing some of these questions when possible."
    )
    return help_text.replace(target, replacement, 1)


class QuestionRoom(PublicMemoryRoom):
    """Question Room for tenured research-question discussion and voting."""

    def __init__(self):
        super().__init__()
        self.room_name = constants.ROOM_QUESTION

    def _get_capsule_type(self) -> str:
        return constants.CAPSULE_TYPE_QUESTION

    def _question_room_key(self, room_context: RoomContext) -> str:
        return room_context.constants_module.SHORT_ROOM_NAME_QUESTION

    def _get_additional_yaml_fields_for_create(self) -> List[str]:
        return [constants.YAML_CAPSULE_ABSTRACT]

    def _is_allowed_agent(self, agent_data: Dict[str, Any], room_context: RoomContext) -> bool:
        station = getattr(room_context, "station_instance", None)
        if station and hasattr(station, "_is_agent_question_room_allowed"):
            return station._is_agent_question_room_allowed(agent_data, station._get_current_tick())

        consts = room_context.constants_module
        if agent_data.get(consts.AGENT_STATUS_KEY) != consts.AGENT_STATUS_RECURSIVE:
            return False
        return supervisor_utils.is_supervisor(agent_data, consts)

    def _normalize_question_capsule(self, capsule_data: Dict[str, Any]) -> bool:
        changed = False
        status = str(capsule_data.get(constants.QUESTION_STATUS_KEY) or "").strip().lower()
        valid_statuses = {
            constants.QUESTION_STATUS_PENDING,
            constants.QUESTION_STATUS_OPEN,
            constants.QUESTION_STATUS_REDACTED,
            constants.QUESTION_STATUS_SOLVED,
            constants.QUESTION_STATUS_RETIRED,
        }
        if status not in valid_statuses:
            capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_PENDING
            changed = True

        votes = capsule_data.get(constants.QUESTION_VOTES_KEY)
        if not isinstance(votes, dict):
            votes = {}
            capsule_data[constants.QUESTION_VOTES_KEY] = votes
            changed = True
        cleaned_votes = {
            str(agent_name): 1 if int(value) > 0 else -1
            for agent_name, value in votes.items()
            if str(agent_name).strip() and str(value).strip() in {"1", "-1"}
        }
        if cleaned_votes != votes:
            capsule_data[constants.QUESTION_VOTES_KEY] = cleaned_votes
            changed = True

        net_upvote = sum(cleaned_votes.values())
        if capsule_data.get(constants.QUESTION_NET_UPVOTE_KEY) != net_upvote:
            capsule_data[constants.QUESTION_NET_UPVOTE_KEY] = net_upvote
            changed = True

        if constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY not in capsule_data:
            capsule_data[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = None
            changed = True

        for message in capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []):
            if not isinstance(message, dict):
                continue
            solution_votes = message.get(constants.QUESTION_SOLUTION_VOTES_KEY)
            if not isinstance(solution_votes, dict):
                solution_votes = {}
                message[constants.QUESTION_SOLUTION_VOTES_KEY] = solution_votes
                changed = True
            cleaned_solution_votes = {
                str(agent_name): 1 if int(value) > 0 else -1
                for agent_name, value in solution_votes.items()
                if str(agent_name).strip() and str(value).strip() in {"1", "-1"}
            }
            if cleaned_solution_votes != solution_votes:
                message[constants.QUESTION_SOLUTION_VOTES_KEY] = cleaned_solution_votes
                changed = True
            solution_net = sum(cleaned_solution_votes.values())
            if message.get(constants.QUESTION_SOLUTION_NET_UPVOTE_KEY) != solution_net:
                message[constants.QUESTION_SOLUTION_NET_UPVOTE_KEY] = solution_net
                changed = True

        return changed

    def _save_question_capsule(self, capsule_data: Dict[str, Any], numeric_id: int) -> None:
        path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.CAPSULES_DIR_NAME,
            constants.QUESTION_CAPSULES_SUBDIR_NAME,
            f"question_{numeric_id}{constants.YAML_EXTENSION}",
        )
        file_io_utils.save_yaml(capsule_data, path)
        capsule_manager._sync_capsule_index_after_save(capsule_data, path, None)

    def _get_voting_obligation_footer(
        self,
        capsule_data: Dict[str, Any],
        agent_data: Dict[str, Any],
    ) -> Optional[str]:
        agent_name = agent_data.get(constants.AGENT_NAME_KEY)
        if not agent_name:
            return None
        if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY):
            return None
        if capsule_data.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING) in {
            constants.QUESTION_STATUS_SOLVED,
            constants.QUESTION_STATUS_RETIRED,
        }:
            return None
        votes = capsule_data.get(constants.QUESTION_VOTES_KEY)
        can_vote_on_question = (
            capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY) != agent_name
            and not (isinstance(votes, dict) and agent_name in votes)
        )
        if can_vote_on_question:
            return _QUESTION_AND_SOLUTION_VOTING_OBLIGATION_FOOTER
        return _SOLUTION_VOTING_OBLIGATION_FOOTER

    def _refresh_solution_status(self, capsule_data: Dict[str, Any]) -> None:
        if capsule_data.get(constants.QUESTION_STATUS_KEY) == constants.QUESTION_STATUS_RETIRED:
            return
        accepted_message: Optional[Dict[str, Any]] = None
        accepted_net = 3
        problem_author = capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY)
        for message in capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []):
            if not isinstance(message, dict) or message.get(constants.MESSAGE_IS_DELETED_KEY):
                continue
            if message.get(constants.MESSAGE_AUTHOR_NAME_KEY) == problem_author:
                continue
            solution_net = int(message.get(constants.QUESTION_SOLUTION_NET_UPVOTE_KEY, 0) or 0)
            if solution_net >= accepted_net:
                accepted_message = message
                accepted_net = solution_net

        if accepted_message:
            capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_SOLVED
            capsule_data[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = accepted_message.get(constants.MESSAGE_ID_KEY)
            return

        if capsule_data.get(constants.QUESTION_STATUS_KEY) == constants.QUESTION_STATUS_SOLVED:
            capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_OPEN
        capsule_data[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = None

    def _notify_vote_author(
        self,
        author_name: Optional[str],
        voter_name: str,
        item_kind: str,
        display_id: str,
        vote_value: int,
        net_upvote: int,
        room_context: RoomContext,
    ) -> None:
        if not author_name or author_name == voter_name:
            return

        consts = room_context.constants_module
        vote_word = "upvoted" if vote_value > 0 else "downvoted"
        notification_text = (
            f"Your {item_kind} (Question #{display_id}) has been {vote_word} by {voter_name}; "
            f"current total net upvote is {net_upvote}."
        )

        def update_author(author_agent_data: Dict[str, Any]) -> None:
            if not self._is_allowed_agent(author_agent_data, room_context):
                return
            if author_agent_data.get(consts.AGENT_SESSION_ENDED_KEY) or author_agent_data.get(consts.AGENT_IS_ASCENDED_KEY):
                return
            room_context.agent_manager.add_pending_notification(author_agent_data, notification_text)

        try:
            room_context.agent_manager.update_agent_with_function(author_name, update_author)
        except Exception as exc:
            print(f"QuestionRoom: failed to notify {author_name} about vote on Question #{display_id}: {exc}")

    def _active_question_count_for_author(self, author_name: str) -> int:
        count = 0
        for meta in capsule_manager.list_capsules(constants.CAPSULE_TYPE_QUESTION, None):
            if (
                meta.get(constants.CAPSULE_AUTHOR_NAME_KEY) == author_name
                and meta.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING)
                != constants.QUESTION_STATUS_RETIRED
            ):
                count += 1
        return count

    def _get_question_filter(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        return room_context.agent_manager.get_agent_room_state(
            agent_data,
            self._question_room_key(room_context),
            "question_filter_status",
            default=None,
        )

    def _set_question_filter(self, agent_data: Dict[str, Any], room_context: RoomContext, status: Optional[str]) -> None:
        room_context.agent_manager.set_agent_room_state(
            agent_data,
            self._question_room_key(room_context),
            "question_filter_status",
            status,
        )

    def _get_question_rank(self, agent_data: Dict[str, Any], room_context: RoomContext) -> Optional[str]:
        return room_context.agent_manager.get_agent_room_state(
            agent_data,
            self._question_room_key(room_context),
            "question_rank",
            default=None,
        )

    def _set_question_rank(self, agent_data: Dict[str, Any], room_context: RoomContext, rank: Optional[str]) -> None:
        room_context.agent_manager.set_agent_room_state(
            agent_data,
            self._question_room_key(room_context),
            "question_rank",
            rank,
        )

    def _format_solution_message_ref(self, message_id: Optional[Any]) -> Optional[str]:
        if not message_id:
            return None
        message_id_str = str(message_id).strip()
        if not message_id_str:
            return None
        match = re.fullmatch(r"(?:question_)?(\d+)-(\d+)", message_id_str)
        if match:
            return f"#{match.group(1)}-{match.group(2)}"
        return message_id_str

    def _format_capsule_for_list_display(self, capsule_metadata: Dict[str, Any], agent_read_status: Dict[str, bool], room_context: RoomContext) -> str:
        consts = room_context.constants_module
        capsule_id_str = capsule_metadata.get(consts.CAPSULE_ID_KEY, "N/A")
        match = re.search(r"(\d+)$", str(capsule_id_str))
        numeric_id_part = match.group(1) if match else str(capsule_id_str)
        title = str(capsule_metadata.get(consts.CAPSULE_TITLE_KEY, "No Title")).replace("|", "&#124;")
        author = str(capsule_metadata.get(consts.CAPSULE_AUTHOR_NAME_KEY, "Unknown")).replace("|", "&#124;")
        date_tick = capsule_metadata.get(consts.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
        word_count = capsule_metadata.get(consts.CAPSULE_WORD_COUNT_TOTAL_KEY, 0)
        total_messages = capsule_metadata.get("total_message_count", 0)
        unread_count = capsule_metadata.get(consts.CAPSULE_UNREAD_MESSAGE_COUNT_KEY, 0)
        if total_messages == 0:
            read_str = "(No Msgs)"
        elif unread_count > 0:
            read_str = f"({unread_count} unread)"
        else:
            read_str = "(All Read)"
        question_status = capsule_metadata.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING)
        status_display = str(question_status)
        solved_by_ref = self._format_solution_message_ref(capsule_metadata.get(consts.QUESTION_SOLVED_BY_MESSAGE_ID_KEY))
        if question_status == consts.QUESTION_STATUS_SOLVED and solved_by_ref:
            status_display = f"{status_display} ({solved_by_ref})"
        status_display = status_display.replace("|", "&#124;")
        net_upvote = capsule_metadata.get(consts.QUESTION_NET_UPVOTE_KEY, 0)
        return (
            f"| {numeric_id_part} | {title} | {author} | Tick {date_tick} | {word_count} | "
            f"{total_messages} | {status_display} | {net_upvote} | {read_str} |"
        )

    def _format_question_status_summary(self, capsule_metadata: List[Dict[str, Any]], consts: Any) -> str:
        counts = {
            consts.QUESTION_STATUS_OPEN: 0,
            consts.QUESTION_STATUS_PENDING: 0,
            consts.QUESTION_STATUS_SOLVED: 0,
            consts.QUESTION_STATUS_RETIRED: 0,
        }
        for capsule_meta in capsule_metadata:
            question_status = capsule_meta.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING)
            if question_status in counts:
                counts[question_status] += 1
        return (
            f"Current number of open questions: {counts[consts.QUESTION_STATUS_OPEN]}; "
            f"pending questions: {counts[consts.QUESTION_STATUS_PENDING]}; "
            f"solved questions: {counts[consts.QUESTION_STATUS_SOLVED]}; "
            f"retired questions: {counts[consts.QUESTION_STATUS_RETIRED]}."
        )

    def _get_specific_room_content(self,
                                   agent_data: Dict[str, Any],
                                   room_context: RoomContext,
                                   current_tick: int) -> str:
        if not self._is_allowed_agent(agent_data, room_context):
            return "The Question Room is only accessible to tenured Recursive Agents and Supervisors."

        consts = room_context.constants_module
        agent_read_status = self._get_agent_read_status(agent_data, room_context)
        current_page = self._get_current_page(agent_data, room_context)
        page_size = consts.DEFAULT_PAGE_SIZE_CAPSULES
        status_filter = self._get_question_filter(agent_data, room_context)
        rank_mode = self._get_question_rank(agent_data, room_context)
        pinned_ids_full = self._get_pinned_capsules_ids(agent_data, room_context)

        all_capsules = capsule_manager.list_capsules(
            consts.CAPSULE_TYPE_QUESTION,
            None,
            agent_read_status=agent_read_status,
        )
        status_summary = self._format_question_status_summary(all_capsules, consts)
        if status_filter:
            all_capsules = [
                capsule for capsule in all_capsules
                if capsule.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING) == status_filter
            ]
        else:
            all_capsules = [
                capsule for capsule in all_capsules
                if capsule.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING)
                != consts.QUESTION_STATUS_RETIRED
            ]

        pinned_lookup = set(pinned_ids_full)
        pinned_capsules = [capsule for capsule in all_capsules if capsule.get(consts.CAPSULE_ID_KEY) in pinned_lookup]
        capsules_for_page = [capsule for capsule in all_capsules if capsule.get(consts.CAPSULE_ID_KEY) not in pinned_lookup]

        if rank_mode == "upvote":
            sort_key = lambda item: (item.get(consts.QUESTION_NET_UPVOTE_KEY, 0), item.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0))
            pinned_capsules.sort(key=sort_key, reverse=True)
            capsules_for_page.sort(key=sort_key, reverse=True)
        else:
            pinned_capsules.sort(key=lambda item: item.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0), reverse=True)
            capsules_for_page.sort(key=lambda item: item.get(consts.CAPSULE_CREATED_AT_TICK_KEY, 0), reverse=True)

        total_items = len(capsules_for_page)
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 1
        if total_pages <= 0:
            total_pages = 1
        current_page = max(1, min(current_page, total_pages))
        self._set_current_page(agent_data, current_page, room_context)
        start = (current_page - 1) * page_size
        paginated_capsules = capsules_for_page[start:start + page_size]

        table_header = "| ID | Title | Author | Date | Words | Msgs | Status | Net Upvote | Read |"
        table_separator = "|:----|:-------------------------------|:----------------|:----------|:------:|:-----:|:---------|----------:|:---------|"
        output_lines = []

        if status_filter or rank_mode:
            controls = []
            if status_filter:
                controls.append(f"filter: {status_filter}")
            if rank_mode:
                controls.append(f"rank: {rank_mode}")
            output_lines.append("**Question View:** " + ", ".join(controls))
            output_lines.append("")

        if pinned_capsules:
            output_lines.append("**Pinned Capsules**")
            output_lines.append(table_header)
            output_lines.append(table_separator)
            for capsule_meta in pinned_capsules:
                output_lines.append(self._format_capsule_for_list_display(capsule_meta, agent_read_status, room_context))
            output_lines.append("")

        output_lines.append(f"**List of Questions (Page {current_page} / {total_pages})**")
        if paginated_capsules:
            output_lines.append(table_header)
            output_lines.append(table_separator)
            for capsule_meta in paginated_capsules:
                output_lines.append(self._format_capsule_for_list_display(capsule_meta, agent_read_status, room_context))
        else:
            output_lines.append("No questions to display on this page." if total_items > 0 else "No questions available in this room.")
        if total_pages > 1:
            output_lines.append("")
            output_lines.append(f"(Use `/execute_action{{page N}}` to navigate between pages 1-{total_pages}.)")
        output_lines.append("")
        output_lines.append(status_summary)
        return "\n".join(output_lines)

    def _build_question_read_content(
        self,
        capsule_data: Dict[str, Any],
        target_cap_num_id: int,
        agent_read_statuses: Dict[str, bool],
        all_items_marked_read: set,
        target_full_msg_id: Optional[str] = None,
    ) -> List[str]:
        consts = constants
        lines: List[str] = []
        status = capsule_data.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING)
        net_upvote = capsule_data.get(consts.QUESTION_NET_UPVOTE_KEY, 0)
        solved_by = capsule_data.get(consts.QUESTION_SOLVED_BY_MESSAGE_ID_KEY)
        solved_by_ref = self._format_solution_message_ref(solved_by)

        if target_full_msg_id:
            message = next(
                (msg for msg in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, []) if msg.get(consts.MESSAGE_ID_KEY) == target_full_msg_id),
                None,
            )
            if not message:
                return [f"Message {target_full_msg_id} not found in question #{target_cap_num_id}."]
            accepted_marker = " [Accepted Solution]" if target_full_msg_id == solved_by else ""
            lines.append(f"**Message {target_full_msg_id}{accepted_marker} from Question '{capsule_data[consts.CAPSULE_TITLE_KEY]}':**")
            lines.append(f"Question Status: {status}, Question Net Upvote: {net_upvote}")
            if solved_by_ref:
                lines.append(f"Solved By: {solved_by_ref}")
            solution_net = message.get(consts.QUESTION_SOLUTION_NET_UPVOTE_KEY, 0)
            lines.append(f"Author: {message[consts.MESSAGE_AUTHOR_NAME_KEY]} (Tick {message[consts.MESSAGE_POSTED_AT_TICK_KEY]}), Solution Net Upvote: {solution_net}")
            if message.get(consts.MESSAGE_TITLE_KEY):
                lines.append(f"Title: {message[consts.MESSAGE_TITLE_KEY]}")
            lines.append(f"Content:\n{message[consts.MESSAGE_CONTENT_KEY]}")
            all_items_marked_read.add(target_full_msg_id)
            return lines

        lines.append(f"**Question #{target_cap_num_id}: {capsule_data[consts.CAPSULE_TITLE_KEY]}**")
        lines.append(f"Author: {capsule_data[consts.CAPSULE_AUTHOR_NAME_KEY]}, Created at Tick: {capsule_data[consts.CAPSULE_CREATED_AT_TICK_KEY]}")
        lines.append(f"Status: {status}, Net Upvote: {net_upvote}")
        if solved_by_ref:
            lines.append(f"Solved By: {solved_by_ref}")
        if capsule_data.get(consts.CAPSULE_ABSTRACT_KEY):
            lines.append(f"Abstract: {capsule_data[consts.CAPSULE_ABSTRACT_KEY]}")

        messages = capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
        if not messages:
            lines.append("(This question has no messages.)")
        else:
            lines.append("\n**Messages:**")

        for message in messages:
            msg_id_full = message[consts.MESSAGE_ID_KEY]
            msg_index = msg_id_full.split("-")[-1]
            user_friendly_msg_id = f"{target_cap_num_id}-{msg_index}"
            solution_net = message.get(consts.QUESTION_SOLUTION_NET_UPVOTE_KEY, 0)
            accepted_marker = " [Accepted Solution]" if msg_id_full == solved_by else ""
            if agent_read_statuses.get(msg_id_full, False):
                lines.append(
                    f"\n---\n**Message {msg_id_full}{accepted_marker}** "
                    f"(Solution Net Upvote: {solution_net}; already read. Use `/execute_action{{unread {user_friendly_msg_id}}}` to show again.)"
                )
            else:
                lines.append(
                    f"\n---\n**Message {msg_id_full}{accepted_marker}**\n"
                    f"Author: {message[consts.MESSAGE_AUTHOR_NAME_KEY]} "
                    f"(Tick {message[consts.MESSAGE_POSTED_AT_TICK_KEY]}), Solution Net Upvote: {solution_net}"
                )
                if message.get(consts.MESSAGE_TITLE_KEY):
                    lines.append(f"Title: {message[consts.MESSAGE_TITLE_KEY]}")
                lines.append(f"Content:\n{message[consts.MESSAGE_CONTENT_KEY]}")
                all_items_marked_read.add(msg_id_full)
        all_items_marked_read.add(capsule_data[consts.CAPSULE_ID_KEY])
        return lines

    def _handle_question_read(
        self,
        agent_data: Dict[str, Any],
        action_args: Optional[str],
        room_context: RoomContext,
    ) -> Tuple[List[str], Optional[Any]]:
        actions_executed: List[str] = []
        if not action_args:
            actions_executed.append(f"Usage: /execute_action{{{constants.ACTION_CAPSULE_READ} <id1,id2-msg,...>}}")
            return actions_executed, None

        def expand_range(raw_id: str) -> Tuple[List[str], Optional[str]]:
            if ":" not in raw_id:
                return [raw_id], None
            start_part, end_part = [part.strip() for part in raw_id.split(":", 1)]
            try:
                start_has_msg = "-" in start_part
                end_has_msg = "-" in end_part
                if start_has_msg != end_has_msg:
                    return [], "Range should be cross message or cross capsule, not mixture of both"
                if start_has_msg:
                    start_cap, start_msg = [int(part) for part in start_part.split("-", 1)]
                    end_cap, end_msg = [int(part) for part in end_part.split("-", 1)]
                    if start_cap != end_cap:
                        return [], "Range should be cross message or cross capsule, not mixture of both"
                    if start_msg > end_msg:
                        return [], f"Invalid range: {start_msg} > {end_msg} (start > end)"
                    return [f"{start_cap}-{msg}" for msg in range(start_msg, end_msg + 1)], None
                start_cap = int(start_part)
                end_cap = int(end_part)
                if start_cap > end_cap:
                    return [], f"Invalid range: {start_cap} > {end_cap} (start > end)"
                return [str(cap_id) for cap_id in range(start_cap, end_cap + 1)], None
            except ValueError:
                return [], f"Invalid range format: '{raw_id}'"

        expanded_ids: List[str] = []
        for raw_id in [item.strip() for item in action_args.split(",") if item.strip()]:
            ids, error = expand_range(raw_id)
            if error:
                actions_executed.append(f"Read error: {error}")
                return actions_executed, None
            expanded_ids.extend(ids)

        all_content_parts: List[str] = []
        all_items_marked_read: set = set()
        processed_ids: List[str] = []
        voting_obligation_footer: Optional[str] = None
        for target in expanded_ids:
            parts = target.split("-", 1)
            try:
                numeric_id = int(parts[0])
            except ValueError:
                all_content_parts.append(f"Invalid ID format: '{target}'.")
                processed_ids.append(f"{target} (invalid)")
                continue
            full_msg_id = f"question_{numeric_id}-{parts[1]}" if len(parts) > 1 else None
            capsule_data = capsule_manager.get_capsule(
                numeric_id,
                constants.CAPSULE_TYPE_QUESTION,
                None,
                include_deleted_messages=False,
            )
            if not self._check_action_permission(constants.ACTION_CAPSULE_READ, agent_data, room_context, capsule_data, target_numeric_id=numeric_id):
                all_content_parts.append(f"Read failed: Permission denied or item '{target}' not found.")
                processed_ids.append(f"{target} (no access/found)")
                continue
            if not capsule_data:
                all_content_parts.append(f"Question '{target}' not found.")
                processed_ids.append(f"{target} (not found)")
                continue
            self._normalize_question_capsule(capsule_data)
            candidate_footer = self._get_voting_obligation_footer(capsule_data, agent_data)
            if candidate_footer == _QUESTION_AND_SOLUTION_VOTING_OBLIGATION_FOOTER:
                voting_obligation_footer = candidate_footer
            elif voting_obligation_footer is None:
                voting_obligation_footer = candidate_footer
            all_content_parts.extend(
                self._build_question_read_content(
                    capsule_data,
                    numeric_id,
                    self._get_agent_read_status(agent_data, room_context),
                    all_items_marked_read,
                    target_full_msg_id=full_msg_id,
                )
            )
            processed_ids.append(target)

        if voting_obligation_footer:
            all_content_parts.append(voting_obligation_footer)
        if all_content_parts:
            room_context.agent_manager.add_pending_notification(agent_data, "\n\n---\n\n".join(all_content_parts))
        for item_id in all_items_marked_read:
            self._set_agent_read_status(agent_data, item_id, True, room_context)
        actions_executed.append(
            f"Read command processed for: {', '.join(processed_ids)}. Content/status sent to System Messages."
            if processed_ids else "Read command could not be processed for the given IDs."
        )
        return actions_executed, None

    def _check_action_permission(self,
                                 action_command: str,
                                 agent_data: Dict[str, Any],
                                 room_context: RoomContext,
                                 capsule_data: Optional[Dict[str, Any]] = None,
                                 target_id_str: Optional[str] = None,
                                 target_numeric_id: Optional[int] = None) -> bool:
        if not self._is_allowed_agent(agent_data, room_context):
            return False

        consts = room_context.constants_module
        agent_current_lineage = agent_data.get(consts.AGENT_LINEAGE_KEY)
        read_like_actions = [
            consts.ACTION_CAPSULE_READ, consts.ACTION_CAPSULE_PREVIEW,
            consts.ACTION_CAPSULE_PIN, consts.ACTION_CAPSULE_UNPIN,
            consts.ACTION_CAPSULE_SEARCH, consts.ACTION_CAPSULE_PAGE,
            consts.ACTION_CAPSULE_UNREAD, consts.ACTION_CAPSULE_MUTE,
            consts.ACTION_CAPSULE_UNMUTE, consts.ACTION_QUESTION_FILTER,
            consts.ACTION_QUESTION_UNFILTER, consts.ACTION_QUESTION_RANK,
        ]
        if action_command in read_like_actions:
            return True

        if action_command == consts.ACTION_CAPSULE_CREATE:
            author_name = agent_data.get(consts.AGENT_NAME_KEY)
            limit = 3 if supervisor_utils.is_supervisor(agent_data, consts) else 1
            return bool(author_name) and self._active_question_count_for_author(author_name) < limit

        if action_command == consts.ACTION_CAPSULE_REPLY:
            if not capsule_data or capsule_data.get(consts.CAPSULE_IS_DELETED_KEY):
                return False
            status = capsule_data.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING)
            return status in {consts.QUESTION_STATUS_OPEN, consts.QUESTION_STATUS_SOLVED}

        if action_command in [consts.ACTION_QUESTION_UPVOTE, consts.ACTION_QUESTION_DOWNVOTE]:
            return (
                bool(capsule_data)
                and not capsule_data.get(consts.CAPSULE_IS_DELETED_KEY)
                and capsule_data.get(consts.QUESTION_STATUS_KEY, consts.QUESTION_STATUS_PENDING)
                != consts.QUESTION_STATUS_RETIRED
            )

        if action_command in [consts.ACTION_QUESTION_RETIRE, consts.ACTION_QUESTION_UNRETIRE]:
            return (
                bool(capsule_data)
                and not capsule_data.get(consts.CAPSULE_IS_DELETED_KEY)
                and supervisor_utils.is_supervisor(agent_data, consts)
            )

        if action_command in [consts.ACTION_CAPSULE_DELETE, consts.ACTION_CAPSULE_UPDATE]:
            if not capsule_data:
                return False
            if capsule_data.get(consts.CAPSULE_IS_DELETED_KEY) and action_command == consts.ACTION_CAPSULE_UPDATE:
                return False
            item_author_lineage: Optional[str] = None
            if target_id_str and "-" in target_id_str:
                msg_to_check = next(
                    (m for m in capsule_data.get(consts.CAPSULE_MESSAGES_KEY, []) if m.get(consts.MESSAGE_ID_KEY) == target_id_str),
                    None,
                )
                if not msg_to_check:
                    return False
                item_author_lineage = msg_to_check.get(consts.MESSAGE_AUTHOR_LINEAGE_KEY)
            else:
                item_author_lineage = capsule_data.get(consts.CAPSULE_AUTHOR_LINEAGE_KEY)
            return bool(agent_current_lineage and item_author_lineage and agent_current_lineage == item_author_lineage)

        return False

    def _handle_question_retirement(
        self,
        agent_data: Dict[str, Any],
        action_command: str,
        action_args: Optional[str],
        yaml_data: Optional[Dict[str, Any]],
        room_context: RoomContext,
        current_tick: int,
    ) -> Tuple[List[str], Optional[Any]]:
        if not action_args or not str(action_args).strip().isdigit():
            return [f"Usage: /execute_action{{{action_command} <question_id>}}"], None

        numeric_id = int(str(action_args).strip())
        capsule_data = capsule_manager.get_capsule(
            numeric_id,
            constants.CAPSULE_TYPE_QUESTION,
            None,
            include_deleted_capsule=True,
            include_deleted_messages=True,
        )
        if not self._check_action_permission(
            action_command,
            agent_data,
            room_context,
            capsule_data,
            target_numeric_id=numeric_id,
        ):
            return [f"Permission denied or question #{numeric_id} not found for {action_command}."], None

        self._normalize_question_capsule(capsule_data)
        status = capsule_data.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING)
        if action_command == constants.ACTION_QUESTION_RETIRE:
            if status == constants.QUESTION_STATUS_RETIRED:
                return [f"Question #{numeric_id} is already retired."], None

            rationale = yaml_data.get(constants.YAML_CAPSULE_CONTENT) if isinstance(yaml_data, dict) else None
            if isinstance(rationale, str) and rationale.strip():
                original_capsule = capsule_data
                reply_added = capsule_manager.add_message_to_capsule(
                    numeric_id,
                    constants.CAPSULE_TYPE_QUESTION,
                    {constants.YAML_CAPSULE_CONTENT: rationale},
                    agent_data,
                    current_tick,
                    None,
                )
                if not reply_added:
                    return [f"Unable to record the retirement reason for question #{numeric_id}; question was not retired."], None
                capsule_data = capsule_manager.get_capsule(
                    numeric_id,
                    constants.CAPSULE_TYPE_QUESTION,
                    None,
                    include_deleted_capsule=True,
                    include_deleted_messages=True,
                )
                if not capsule_data or not capsule_data.get(constants.CAPSULE_MESSAGES_KEY):
                    return [f"Retirement reason was recorded for question #{numeric_id}, but the question could not be reloaded and was not retired."], None
                new_message = capsule_data[constants.CAPSULE_MESSAGES_KEY][-1]
                self._set_agent_read_status(agent_data, new_message[constants.MESSAGE_ID_KEY], True, room_context)
                self._after_reply_added(original_capsule, new_message, agent_data, room_context, current_tick)

            capsule_data[constants.QUESTION_STATUS_BEFORE_RETIREMENT_KEY] = status
            capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_RETIRED
            capsule_data[constants.QUESTION_RETIRED_BY_KEY] = agent_data.get(constants.AGENT_NAME_KEY)
            capsule_data[constants.QUESTION_RETIRED_AT_TICK_KEY] = current_tick
            result = f"Question #{numeric_id} retired from status {status}."
        else:
            if status != constants.QUESTION_STATUS_RETIRED:
                return [f"Question #{numeric_id} is not retired."], None
            previous_status = capsule_data.get(
                constants.QUESTION_STATUS_BEFORE_RETIREMENT_KEY,
                constants.QUESTION_STATUS_PENDING,
            )
            restorable_statuses = {
                constants.QUESTION_STATUS_PENDING,
                constants.QUESTION_STATUS_OPEN,
                constants.QUESTION_STATUS_REDACTED,
                constants.QUESTION_STATUS_SOLVED,
            }
            if previous_status not in restorable_statuses:
                previous_status = constants.QUESTION_STATUS_PENDING
            capsule_data[constants.QUESTION_STATUS_KEY] = previous_status
            result = f"Question #{numeric_id} unretired and restored to status {previous_status}."

        capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
        self._save_question_capsule(capsule_data, numeric_id)
        return [result], None

    def _handle_question_vote(
        self,
        agent_data: Dict[str, Any],
        action_command: str,
        action_args: Optional[str],
        room_context: RoomContext,
        current_tick: int,
    ) -> Tuple[List[str], Optional[Any]]:
        actions_executed: List[str] = []
        if not action_args:
            actions_executed.append(f"Usage: /execute_action{{{action_command} <question_id or question_id-message_id>}}")
            return actions_executed, None
        target = action_args.strip()
        parts = target.split("-", 1)
        try:
            numeric_id = int(parts[0])
        except ValueError:
            actions_executed.append(f"Invalid question ID for {action_command}: {target}")
            return actions_executed, None

        capsule_data = capsule_manager.get_capsule(
            numeric_id,
            constants.CAPSULE_TYPE_QUESTION,
            None,
            include_deleted_capsule=True,
            include_deleted_messages=True,
        )
        if not self._check_action_permission(action_command, agent_data, room_context, capsule_data, target_numeric_id=numeric_id):
            actions_executed.append(f"Permission denied or question #{numeric_id} not found for {action_command}.")
            return actions_executed, None
        if not capsule_data:
            actions_executed.append(f"Unable to find question #{numeric_id}.")
            return actions_executed, None
        self._normalize_question_capsule(capsule_data)

        vote_value = 1 if action_command == constants.ACTION_QUESTION_UPVOTE else -1
        agent_name = agent_data.get(constants.AGENT_NAME_KEY)
        if not agent_name:
            actions_executed.append("Unable to determine voter name.")
            return actions_executed, None

        if len(parts) == 1:
            if capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY) == agent_name:
                actions_executed.append("You cannot vote on your own question.")
                return actions_executed, None
            votes = capsule_data.setdefault(constants.QUESTION_VOTES_KEY, {})
            votes[agent_name] = vote_value
            net_upvote = sum(int(value) for value in votes.values())
            capsule_data[constants.QUESTION_NET_UPVOTE_KEY] = net_upvote
            previous_status = capsule_data.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING)
            if previous_status == constants.QUESTION_STATUS_PENDING:
                if net_upvote >= 3:
                    capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_OPEN
                elif net_upvote <= -3:
                    capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_REDACTED
            capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
            self._save_question_capsule(capsule_data, numeric_id)
            self._notify_vote_author(
                capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY),
                agent_name,
                "question",
                str(numeric_id),
                vote_value,
                net_upvote,
                room_context,
            )
            new_status = capsule_data.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING)
            status_note = f" Status changed from {previous_status} to {new_status}." if new_status != previous_status else ""
            actions_executed.append(f"Vote recorded for question #{numeric_id}. Net upvote: {net_upvote}.{status_note}")
            return actions_executed, None

        message_id = f"question_{numeric_id}-{parts[1]}"
        status = capsule_data.get(constants.QUESTION_STATUS_KEY, constants.QUESTION_STATUS_PENDING)
        if status not in {constants.QUESTION_STATUS_OPEN, constants.QUESTION_STATUS_SOLVED}:
            actions_executed.append(f"Solution votes are only available on open or solved questions. Question #{numeric_id} is {status}.")
            return actions_executed, None

        problem_author = capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY)
        target_message = next(
            (message for message in capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []) if message.get(constants.MESSAGE_ID_KEY) == message_id),
            None,
        )
        if not target_message or target_message.get(constants.MESSAGE_IS_DELETED_KEY):
            actions_executed.append(f"Unable to find active message {target}.")
            return actions_executed, None
        if target_message.get(constants.MESSAGE_AUTHOR_NAME_KEY) == agent_name:
            actions_executed.append("You cannot vote on your own reply.")
            return actions_executed, None
        if target_message.get(constants.MESSAGE_AUTHOR_NAME_KEY) == problem_author:
            actions_executed.append("Replies authored by the original problem author cannot receive solution votes.")
            return actions_executed, None

        votes = target_message.setdefault(constants.QUESTION_SOLUTION_VOTES_KEY, {})
        votes[agent_name] = vote_value
        solution_net = sum(int(value) for value in votes.values())
        target_message[constants.QUESTION_SOLUTION_NET_UPVOTE_KEY] = solution_net
        previous_status = status
        self._refresh_solution_status(capsule_data)
        capsule_data[constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY] = current_tick
        self._save_question_capsule(capsule_data, numeric_id)
        self._notify_vote_author(
            target_message.get(constants.MESSAGE_AUTHOR_NAME_KEY),
            agent_name,
            "reply",
            f"{numeric_id}-{parts[1]}",
            vote_value,
            solution_net,
            room_context,
        )
        new_status = capsule_data.get(constants.QUESTION_STATUS_KEY, status)
        status_note = f" Status changed from {previous_status} to {new_status}." if new_status != previous_status else ""
        actions_executed.append(f"Solution vote recorded for message {target}. Net solution upvote: {solution_net}.{status_note}")
        return actions_executed, None

    def handle_action(self,
                      agent_data: Dict[str, Any],
                      action_command: str,
                      action_args: Optional[str],
                      yaml_data: Optional[Dict[str, Any]],
                      room_context: RoomContext,
                      current_tick: int) -> Tuple[List[str], Optional[Any]]:
        consts = room_context.constants_module
        if not self._is_allowed_agent(agent_data, room_context):
            return ["The Question Room is only accessible to tenured Recursive Agents and Supervisors."], None

        if action_command in [consts.ACTION_QUESTION_UPVOTE, consts.ACTION_QUESTION_DOWNVOTE]:
            return self._handle_question_vote(agent_data, action_command, action_args, room_context, current_tick)

        if action_command in [consts.ACTION_QUESTION_RETIRE, consts.ACTION_QUESTION_UNRETIRE]:
            return self._handle_question_retirement(agent_data, action_command, action_args, yaml_data, room_context, current_tick)

        if action_command == consts.ACTION_QUESTION_FILTER:
            status = str(action_args or "").strip().lower()
            valid_statuses = {
                consts.QUESTION_STATUS_PENDING,
                consts.QUESTION_STATUS_OPEN,
                consts.QUESTION_STATUS_REDACTED,
                consts.QUESTION_STATUS_SOLVED,
                consts.QUESTION_STATUS_RETIRED,
            }
            if status not in valid_statuses:
                return ["Usage: /execute_action{filter pending|open|redacted|solved|retired}"], None
            self._set_question_filter(agent_data, room_context, status)
            self._set_current_page(agent_data, 1, room_context)
            return [f"Filtered questions by status: {status}."], None

        if action_command == consts.ACTION_QUESTION_UNFILTER:
            self._set_question_filter(agent_data, room_context, None)
            self._set_current_page(agent_data, 1, room_context)
            return ["Question status filter cleared."], None

        if action_command == consts.ACTION_QUESTION_RANK:
            rank = str(action_args or "").strip().lower()
            if rank not in {"id", "upvote"}:
                return ["Usage: /execute_action{rank id|upvote}"], None
            if rank == "id":
                self._set_question_rank(agent_data, room_context, None)
                self._set_current_page(agent_data, 1, room_context)
                return ["Questions ranked by default ID order."], None
            self._set_question_rank(agent_data, room_context, "upvote")
            self._set_current_page(agent_data, 1, room_context)
            return ["Questions ranked by net upvote."], None

        if action_command == consts.ACTION_CAPSULE_READ:
            return self._handle_question_read(agent_data, action_args, room_context)

        return super().handle_action(agent_data, action_command, action_args, yaml_data, room_context, current_tick)

    def get_help_message(self, agent_data: Dict[str, Any], room_context: RoomContext) -> str:
        override_help = self._load_constant_override(room_context, "help")
        if override_help is not None:
            return override_help
        return _build_question_room_help(room_context.constants_module)

    def _send_mention_notifications(self,
                                  mentioned_agents: List[str],
                                  author_name: str,
                                  content: str,
                                  capsule_data: Dict[str, Any],
                                  room_context: RoomContext):
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager
        capsule_title = capsule_data.get(consts.CAPSULE_TITLE_KEY, "this question")
        full_capsule_id = capsule_data.get(consts.CAPSULE_ID_KEY, "unknown_capsule_id")
        numeric_id_match = re.search(r"(\d+)$", full_capsule_id)
        numeric_id = numeric_id_match.group(1) if numeric_id_match else full_capsule_id
        try:
            all_active_agents = agent_manager.get_all_active_agent_names()
        except AttributeError:
            all_active_agents = []
        agent_name_lookup = {name.lower(): name for name in all_active_agents}
        for mentioned_agent in mentioned_agents:
            actual_agent_name = agent_name_lookup.get(mentioned_agent.lower())
            if not actual_agent_name or actual_agent_name == author_name:
                continue
            notification_text = (
                f"{author_name} mentioned you in question \"{capsule_title}\" (#{numeric_id}):\n"
                f"{content}\n"
                f"To reply, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_QUESTION}}}` then `/execute_action{{reply {numeric_id}}}`.\n"
                f"To mute, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_QUESTION}}}` then `/execute_action{{mute {numeric_id}}}`."
            )
            first_msg_id = f"{full_capsule_id}-1"

            def update_mentioned_agent(mentioned_agent_data: Dict[str, Any]) -> None:
                if not self._is_allowed_agent(mentioned_agent_data, room_context):
                    return
                if mentioned_agent_data.get(consts.AGENT_SESSION_ENDED_KEY) or mentioned_agent_data.get(consts.AGENT_IS_ASCENDED_KEY):
                    return
                room_short_name = consts.SHORT_ROOM_NAME_QUESTION
                if room_short_name not in mentioned_agent_data or not isinstance(mentioned_agent_data.get(room_short_name), dict):
                    mentioned_agent_data[room_short_name] = {}
                muted_capsules = mentioned_agent_data[room_short_name].get(consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, {})
                if muted_capsules.get(full_capsule_id, False):
                    return
                read_status = mentioned_agent_data[room_short_name].get(consts.AGENT_ROOM_STATE_READ_STATUS_KEY)
                if not isinstance(read_status, dict):
                    read_status = {}
                    mentioned_agent_data[room_short_name][consts.AGENT_ROOM_STATE_READ_STATUS_KEY] = read_status
                read_status[first_msg_id] = True
                agent_manager.add_pending_notification(mentioned_agent_data, notification_text)

            agent_manager.update_agent_with_function(actual_agent_name, update_mentioned_agent)

    def _after_capsule_created(self,
                               new_capsule_data: Dict[str, Any],
                               creator_agent_data: Dict[str, Any],
                               room_context: RoomContext,
                               current_tick: int):
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager
        author_name = new_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY, "An unknown agent")
        capsule_title = new_capsule_data.get(consts.CAPSULE_TITLE_KEY, "Untitled Question")
        capsule_content = ""
        messages = new_capsule_data.get(consts.CAPSULE_MESSAGES_KEY, [])
        if messages:
            capsule_content = messages[0].get(consts.MESSAGE_CONTENT_KEY, "")
        full_capsule_id = new_capsule_data.get(consts.CAPSULE_ID_KEY, "question_0")
        numeric_id_match = re.search(r"(\d+)$", full_capsule_id)
        numeric_id = numeric_id_match.group(1) if numeric_id_match else full_capsule_id
        word_count = new_capsule_data.get(consts.CAPSULE_WORD_COUNT_TOTAL_KEY, 0)
        mentioned_agents = self._extract_mentions(capsule_content)
        if mentioned_agents:
            self._send_mention_notifications(mentioned_agents, author_name, capsule_content, new_capsule_data, room_context)
        try:
            all_active_agent_names = agent_manager.get_all_active_agent_names()
        except AttributeError:
            all_active_agent_names = []
        mentioned_lookup = {name.lower() for name in mentioned_agents}
        notification_text = (
            f"A new Question Room problem (#{numeric_id}), titled \"{capsule_title}\", "
            f"has been posted by {author_name} ({word_count} words).\n"
            f"To read it, go to the Question Room using: /execute_action{{goto {consts.SHORT_ROOM_NAME_QUESTION}}} /execute_action{{read {numeric_id}}}."
        )
        for other_agent_name in all_active_agent_names:
            if other_agent_name == author_name or other_agent_name.lower() in mentioned_lookup:
                continue

            def update_other_agent(other_agent_data: Dict[str, Any]) -> None:
                if not self._is_allowed_agent(other_agent_data, room_context):
                    return
                if other_agent_data.get(consts.AGENT_SESSION_ENDED_KEY) or other_agent_data.get(consts.AGENT_IS_ASCENDED_KEY):
                    return
                agent_manager.add_pending_notification(other_agent_data, notification_text)

            agent_manager.update_agent_with_function(other_agent_name, update_other_agent)

    def _after_reply_added(self,
                           target_capsule_data: Dict[str, Any],
                           new_message_data: Dict[str, Any],
                           replier_agent_data: Dict[str, Any],
                           room_context: RoomContext,
                           current_tick: int):
        consts = room_context.constants_module
        agent_manager = room_context.agent_manager
        replier_name = replier_agent_data.get(consts.AGENT_NAME_KEY, "An agent")
        original_capsule_id_full = target_capsule_data.get(consts.CAPSULE_ID_KEY, "unknown_capsule_id")
        numeric_id_match = re.search(r"(\d+)$", original_capsule_id_full)
        numeric_id = numeric_id_match.group(1) if numeric_id_match else original_capsule_id_full
        capsule_title = target_capsule_data.get(consts.CAPSULE_TITLE_KEY, "this question")
        new_msg_id_full = new_message_data.get(consts.MESSAGE_ID_KEY, "new_message")
        new_msg_content = new_message_data.get(consts.MESSAGE_CONTENT_KEY, "")
        mentioned_agents = self._extract_mentions(new_msg_content)
        try:
            all_active_agents = agent_manager.get_active_recursive_agent_names()
            agent_name_lookup = {name.lower(): name for name in all_active_agents}
        except AttributeError:
            agent_name_lookup = {}
        mentioned_names = set()
        for mentioned_agent in mentioned_agents:
            actual_agent_name = agent_name_lookup.get(mentioned_agent.lower())
            if actual_agent_name and actual_agent_name != replier_name:
                mentioned_names.add(actual_agent_name)
        thread_participants = set()
        original_author = target_capsule_data.get(consts.CAPSULE_AUTHOR_NAME_KEY)
        if original_author and original_author != replier_name:
            thread_participants.add(original_author)
        for message in target_capsule_data.get(consts.CAPSULE_MESSAGES_KEY, []):
            if not message.get(consts.MESSAGE_IS_DELETED_KEY, False):
                msg_author = message.get(consts.MESSAGE_AUTHOR_NAME_KEY)
                if msg_author and msg_author != replier_name:
                    thread_participants.add(msg_author)
        for agent_to_notify_name in mentioned_names | thread_participants:
            was_mentioned = agent_to_notify_name in mentioned_names
            notification_text = (
                f"{replier_name} replied to question \"{capsule_title}\" (#{numeric_id}) "
                f"(message #{new_msg_id_full})"
            )
            notification_text += " and mentioned you:\n" if was_mentioned else ":\n"
            notification_text += new_msg_content
            notification_text += (
                f"\nTo reply, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_QUESTION}}}` then `/execute_action{{reply {numeric_id}}}`.\n"
                f"To mute, use `/execute_action{{goto {consts.SHORT_ROOM_NAME_QUESTION}}}` then `/execute_action{{mute {numeric_id}}}`."
            )

            def update_agent_to_notify(agent_to_notify_data: Dict[str, Any]) -> None:
                if not self._is_allowed_agent(agent_to_notify_data, room_context):
                    return
                if agent_to_notify_data.get(consts.AGENT_SESSION_ENDED_KEY) or agent_to_notify_data.get(consts.AGENT_IS_ASCENDED_KEY):
                    return
                room_short_name = consts.SHORT_ROOM_NAME_QUESTION
                if room_short_name not in agent_to_notify_data or not isinstance(agent_to_notify_data.get(room_short_name), dict):
                    agent_to_notify_data[room_short_name] = {}
                muted_capsules = agent_to_notify_data[room_short_name].get(consts.AGENT_ROOM_STATE_MUTED_CAPSULES_KEY, {})
                if muted_capsules.get(original_capsule_id_full, False):
                    return
                read_status = agent_to_notify_data[room_short_name].get(consts.AGENT_ROOM_STATE_READ_STATUS_KEY)
                if not isinstance(read_status, dict):
                    read_status = {}
                    agent_to_notify_data[room_short_name][consts.AGENT_ROOM_STATE_READ_STATUS_KEY] = read_status
                read_status[new_msg_id_full] = True
                agent_manager.add_pending_notification(agent_to_notify_data, notification_text)

            agent_manager.update_agent_with_function(agent_to_notify_name, update_agent_to_notify)
