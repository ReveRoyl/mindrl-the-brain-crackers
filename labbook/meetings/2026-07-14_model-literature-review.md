# Team Meeting: Model Literature Review

- **Date:** 2026-07-14
- **Meeting type:** Literature review and model selection
- **Status:** Completed

## 1. Meeting objectives

- Review cognitive models used in relevant published studies.
- Assess whether these models are compatible with the MindRL dataset.
- Identify candidate models for further discussion and implementation.
- Prepare unresolved questions for a supervisor meeting.

## 2. Papers for review

Team members will review the following papers:

1. [Neuroscience paper on ScienceDirect](http://www.sciencedirect.com/science/article/pii/S0896627311001255)
2. [eLife article: e11305](https://elifesciences.org/content/5/e11305v1)
3. [Nature Mental Health article](https://www.nature.com/articles/s44271-023-00031-y)
4. [PsyArXiv preprint](https://osf.io/preprints/psyarxiv/zp6vk_v1)

## 3. Literature-review task

For each paper, team members should identify:

- the behavioural task studied;
- the model's core cognitive mechanism;
- the model inputs, latent states, and outputs;
- the model parameters and their interpretations;
- the learning or decision rule;
- the evidence used to validate the model;
- whether the model requires information available in the MindRL dataset;
- whether the model could generalise across participants and tasks;
- major assumptions and potential limitations.

## 4. Model suitability criteria

Candidate models should be assessed according to:

- compatibility with the available data;
- cognitive interpretability;
- predictive performance;
- ability to generalise to unseen participants and tasks;
- number and identifiability of parameters;
- computational feasibility;
- consistency with the frozen-parameter evaluation requirement;
- suitability for the Interpretation Card.

## 5. Team communication

When a team member identifies a potentially suitable model, they should update the group with:

- the model name;
- the relevant paper;
- a concise description of the mechanism;
- why it may fit the dataset;
- required variables;
- possible implementation challenges;
- key questions for team discussion.

Candidate models will be discussed collectively before implementation.

## 6. Questions for supervisors

Each team member should summarise unresolved questions concerning:

- task structure and dataset interpretation;
- appropriate model families;
- cross-task generalisation;
- parameter estimation;
- model comparison and validation;
- Interpretation Card expectations.

## 7. Action items

| Task | Owner | Status |
|---|---|---|
| Review assigned papers | All members | In progress |
| Identify candidate cognitive models | All members | In progress |
| Share suitable models in the group | All members | Ongoing |
| Summarise unresolved questions | All members | To do |
| Arrange supervisor meeting for Thursday | Team | Tentative |

## 8. Next step

A supervisor meeting is tentatively planned for Thursday. Before the meeting, the team will consolidate:

- candidate models;
- model-selection criteria;
- dataset-related uncertainties;
- technical and conceptual questions.