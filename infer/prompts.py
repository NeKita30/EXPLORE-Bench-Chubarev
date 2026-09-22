final_scene_prediction_prompt = """
You are given an image showing a initial scene.
Then, a sequence of atomic actions occurs:
'{atomic_actions}'

Notes:
- 'C' refers to the camera wearer (the person who is wearing or holding the camera).
- 'X' refers to someone other than the camera wearer.
- Each atomic action description is separated by the `|` character.
- The atomic actions happen in the given order, from left to right.

Your task:
1. Mentally apply this sequence of atomic actions to the initial scene in the image.
2. Imagine what the final scene would look like after all actions have occurred.
3. Describe the final scene in rich, concrete visual detail, as if you are describing the final image.

Requirements:
- Focus **only** on the final scene after all actions, not the intermediate steps.
- Do **not** explain your reasoning or mention the actions explicitly.
- Do **not** mention that you are imagining or predicting; just describe the final scene directly.
- Provide rich, concrete visual detail: entities, appearances, spatial relationships, and interactions.
- Avoid any meta-commentary (no phrases like "the image would show", "I imagine that", etc.).

Now, describe the final scene in detail.
"""


mutil_scene_prediction_prompt = """
You are given an image showing an initial scene.
Then a sequence of atomic actions occurs, provided in {segment_num} consecutive segments (Segment 1, Segment 2, ..., Segment {segment_num}). The full sequence is continuous in time, and the segments are in chronological order.

Atomic action segments:
{atomic_action_segments}

Notes:
- 'C' refers to the camera wearer (the person wearing/holding the camera).
- 'X' refers to someone other than the camera wearer.
- Within each segment, atomic actions are separated by the `|` character.
- Segments happen in order from Segment 1 to Segment {segment_num}. All actions within a segment happen before the next segment begins.

Your task:
- Starting from the initial scene in the image, mentally apply Segment 1’s actions and describe the resulting scene (Scene 1).
- Then, using your previously described Scene 1 as the current state (grounded by the initial image), mentally apply Segment 2’s actions and describe the resulting scene (Scene 2).
- Continue this process until Segment {segment_num}, producing Scene {segment_num}.
- Describe the scenes in rich, concrete visual detail, as if you are describing the images.

Description requirements for each scene:
- Describe **only** the scene after that segment’s actions are complete (do not describe intermediate steps).
- Do **not** explain your reasoning or mention the actions explicitly.
- Do **not** mention that you are imagining or predicting; describe the scene directly.
- Provide rich, concrete visual detail: entities, appearances, spatial relationships, and interactions.
- Avoid meta-commentary (no phrases like "the image would show", "I imagine that", etc.).

Output format (strict):
- Output a valid JSON list with exactly {segment_num} elements (no nested lists, no extra keys, no extra text outside the list).
- The i-th element must be the scene description after completing Segment i.
- Each element must be a single string enclosed in double quotes.

Now output the list of {segment_num} scene descriptions.
"""


multi_rollout_scene_prediction_prompt_next = """
You are given an image showing the previous initial scene.

You are also given the current scene state description (after previous actions):
"{previous_scene}"

Then, the following atomic actions occur next:
'{atomic_actions}'

Notes:
- 'C' refers to the camera wearer (the person who is wearing or holding the camera).
- 'X' refers to someone other than the camera wearer.
- Each atomic action description is separated by the `|` character.
- The atomic actions happen in the given order, from left to right.

Your task:
1. Treat the provided current scene description as the starting state (grounded by the initial image), mentally apply the new actions.
2. Imagine what the resulting scene would look like after all actions have occurred.
3. Describe the resulting scene in rich, concrete visual detail, as if you are describing the final image.

Requirements:
- Describe **only** the resulting scene after all actions (no intermediate steps).
- Do **not** explain your reasoning or mention the actions explicitly.
- Do **not** mention that you are imagining or predicting; describe the scene directly.
- Provide rich, concrete visual detail: entities, appearances, spatial relationships, and interactions.
- Avoid meta-commentary (no phrases like "the image would show", "I imagine that", etc.).

Now write the resulting scene description as plain text in detail.
"""

my_strategy_draft_prediction_prompt = """
You are given an image showing a initial scene.
Then, a sequence of atomic actions occurs:
'{atomic_actions}'

Notes:
- 'C' refers to the camera wearer (the person who is wearing or holding the camera).
- 'X' refers to someone other than the camera wearer.
- Each atomic action description is separated by the `|` character.
- The atomic actions happen in the given order, from left to right.

Your task:
1. Mentally apply this sequence of atomic actions to the initial scene in the image.
2. Imagine what the final scene would look like after all actions have occurred.
3. Describe the final scene in rich, concrete visual detail, as if you are describing the final image.

Requirements:
- Focus **only** on the final scene after all actions, not the intermediate steps.
- Do **not** explain your reasoning or mention the actions explicitly.
- Do **not** mention that you are imagining or predicting; just describe the final scene directly.
- Provide rich, concrete visual detail: entities, appearances, spatial relationships, and interactions.
- Avoid any meta-commentary (no phrases like "the image would show", "I imagine that", etc.).

Now, describe the final scene in detail.
"""

my_strategy_review_prediction_prompt = """
You are given an image showing the initial scene.

Then, a sequence of atomic actions occurs:
'{atomic_actions}'

Notes:
- 'C' refers to the camera wearer (the person who is wearing or holding the camera).
- 'X' refers to someone other than the camera wearer.
- Each atomic action description is separated by the `|` character.
- The atomic actions happen in the given order, from left to right.

You are also given the draft of scene state description after all actions:
"{draft_description}"

Your task:
1. Review the scene state description draft as a result of application of actions to the initial scene.
2. Find objects, that are missing or hallucinated in the scene state description; attributes and relations that are incorrect.
3. Produce a correct final scene state description, fixing found errors: add missing objects and remove hallucinated ones, edit incorrect attributes and relations.

Requirements:
- Focus **only** on the final scene after all actions, not the intermediate steps.
- Do **not** explain your reasoning or mention the actions explicitly.
- Do **not** mention that you are imagining or predicting; describe the scene directly.
- Provide rich, concrete visual detail: entities, appearances, spatial relationships, and interactions.
- Avoid meta-commentary (no phrases like "the image would show", "I imagine that", etc.).

Now write the resulting scene description as plain text in detail.
"""