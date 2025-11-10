# Task Reannotation Report

This report summarizes all tasks that were corrected from reward 0 (failed) to non-zero (successful).

---

## gelato-30b

### run_1

**Total corrections:** 14

| Task ID | App | Original Reward | New Reward | Notes |
|---------|-----|-----------------|------------|-------|
| `3720f614-37fd-4d04-8a6b-76f54f8c222d` | chrome | 0 | 1 | - |
| `59155008-fe71-45ec-8a8f-dc35497b6aa8` | chrome | 0 | 1 | - |
| `82bc8d6a-36eb-4d2d-8801-ef714fb1e55a` | chrome | 0 | 1 | - |
| `93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9` | chrome | 0 | 1 | - |
| `9f935cce-0a9f-435f-8007-817732bfc0a5` | chrome | 0 | 1 | - |
| `b7895e80-f4d1-4648-bee0-4eb45a6f1fa8` | chrome | 0 | 1 | Agent did everything correctly (picked New York, right dates, two people, sort by price ascending). |
| `121ba48f-9e17-48ce-9bc6-a4fb17a7ebba` | chrome | 0 | 1 | Agent successfully added Dota 2 all DLC to cart. Even though two DLCs are listed, only one actually shows up in the cart (maybe because the other is free?). When I do the same in the steam store, I get the same result. The agent further opened the cart to verify. There was an SSL error and it pressed reload to successfully view the cart. Then it terminated. |
| `c1fa57f3-c3db-4596-8f09-020701085416` | chrome | 0 | 1 | - |
| `da46d875-6b82-4681-9284-653b0c7ae241` | chrome | 0 | 1 | - |
| `f0b971a1-6831-4b9b-a50e-22a6e47f45ba` | chrome | 0 | 1 | Not clear from instruction if we want the Super Bowl game that happened in 2019 or the one from the 2019 season. |
| `fc6d8143-9452-4171-9459-7f515143419a` | chrome | 0 | 1 | - |
| `841b50aa-df53-47bd-a73a-22d3a9f73160` | libreoffice_impress | 0 | 1 | - |
| `d06f0d4d-2cd5-4ede-8de9-598629438c6e` | vlc | 0 | 1 | The agent downloaded a dark-theme skin from the internet and successfully set it as the VLC media player’s default. The volume slider is black and overall the design is in a dark color scheme. |
| `e2dd0213-26db-4349-abe5-d5667bfd725c` | gimp | 0 | 1 | Task asks agent to move a centered text box to the left without specifying how much the offset should be. Agent moves the text box but not as much as where the evaluation function expects it to be. |

### run_2

**Total corrections:** 10

| Task ID | App | Original Reward | New Reward | Notes |
|---------|-----|-----------------|------------|-------|
| `0d8b7de3-e8de-4d86-b9fd-dd2dce58a217` | chrome | 0 | 1 | - |
| `2888b4e6-5b47-4b57-8bf5-c73827890774` | chrome | 0 | 1 | - |
| `59155008-fe71-45ec-8a8f-dc35497b6aa8` | chrome | 0 | 1 | - |
| `93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9` | chrome | 0 | 1 | - |
| `9f935cce-0a9f-435f-8007-817732bfc0a5` | chrome | 0 | 1 | - |
| `b7895e80-f4d1-4648-bee0-4eb45a6f1fa8` | chrome | 0 | 1 | It successfully filtered for New York, the weekend, 2 guests and sorted by price. It is unclear what is meant by find a hotel and the agent chooses to terminate on the overview page. |
| `b4f95342-463e-4179-8c3f-193cd7241fb2` | chrome | 0 | 1 | - |
| `fc6d8143-9452-4171-9459-7f515143419a` | chrome | 0 | 1 | - |
| `d06f0d4d-2cd5-4ede-8de9-598629438c6e` | vlc | 0 | 1 | The agent downloaded a dark-theme skin from the internet and successfully set it as the VLC media player’s default. The volume slider is black and overall the design is in a dark color scheme. |
| `550ce7e7-747b-495f-b122-acdc4d0b8e54` | libreoffice_impress | 0 | 1 | Agent did put strike-through through top 2 bullet points, did not alter the slide otherwise |


### run_3

**Total corrections:** 10

| Task ID | App | Original Reward | New Reward | Notes |
|---------|-----|-----------------|------------|-------|
| `2888b4e6-5b47-4b57-8bf5-c73827890774` | chrome | 0 | 1 | - |
| `6c4c23a1-42a4-43cc-9db1-2f86ff3738cc` | chrome | 0 | 1 | - |
| `82bc8d6a-36eb-4d2d-8801-ef714fb1e55a` | chrome | 0 | 1 | - |
| `93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9` | chrome | 0 | 1 | - |
| `9f935cce-0a9f-435f-8007-817732bfc0a5` | chrome | 0 | 1 | - |
| `b7895e80-f4d1-4648-bee0-4eb45a6f1fa8` | chrome | 0 | 1 | It successfully filtered for New York, the weekend, 2 guests and sorted by price. It is unclear what is meant by find a hotel and the agent chooses to terminate on the overview page. |
| `121ba48f-9e17-48ce-9bc6-a4fb17a7ebba` | chrome | 0 | 1 | Agent successfully added Dota 2 all DLC to cart. Even though two DLCs are listed, only one actually shows up in the cart (maybe because the other is free?). When I do the same in the steam store, I get the same result. The agent further opened the cart to verify. There was an SSL error and it pressed reload to successfully view the cart. Then it terminated. |
| `b4f95342-463e-4179-8c3f-193cd7241fb2` | chrome | 0 | 1 | - |
| `fc6d8143-9452-4171-9459-7f515143419a` | chrome | 0 | 1 | - |
| `e2dd0213-26db-4349-abe5-d5667bfd725c` | gimp | 0 | 1 | Task asks agent to move a centered text box to the left without specifying how much the offset should be. Agent moves the text box but not as much as where the evaluation function expects it to be. |
---

## gta1-32b_baseline

### run_1

**Total corrections:** 10

| Task ID | App | Original Reward | New Reward | Notes |
|---------|-----|-----------------|------------|-------|
| `06fe7178-4491-4589-810f-2e2bc9502122` | chrome | 0 | 1 | - |
| `0d8b7de3-e8de-4d86-b9fd-dd2dce58a217` | chrome | 0 | 1 | - |
| `3720f614-37fd-4d04-8a6b-76f54f8c222d` | chrome | 0 | 1 | - |
| `59155008-fe71-45ec-8a8f-dc35497b6aa8` | chrome | 0 | 1 | - |
| `6c4c23a1-42a4-43cc-9db1-2f86ff3738cc` | chrome | 0 | 1 | - |
| `82bc8d6a-36eb-4d2d-8801-ef714fb1e55a` | chrome | 0 | 1 | - |
| `121ba48f-9e17-48ce-9bc6-a4fb17a7ebba` | chrome | 0 | 1 | - |
| `93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9` | chrome | 0 | 1 | - |
| `9f935cce-0a9f-435f-8007-817732bfc0a5` | chrome | 0 | 1 | - |
| `d06f0d4d-2cd5-4ede-8de9-598629438c6e` | vlc | 0 | 1 | The agent downloaded a dark-theme skin from the internet and successfully set it as the VLC media player’s default. The volume slider is black and overall the design is in a dark color scheme. |

### run_2

**Total corrections:** 8

| Task ID | App | Original Reward | New Reward | Notes |
|---------|-----|-----------------|------------|-------|
| `0d8b7de3-e8de-4d86-b9fd-dd2dce58a217` | chrome | 0 | 1 | - |
| `59155008-fe71-45ec-8a8f-dc35497b6aa8` | chrome | 0 | 1 | - |
| `82bc8d6a-36eb-4d2d-8801-ef714fb1e55a` | chrome | 0 | 1 | - |
| `121ba48f-9e17-48ce-9bc6-a4fb17a7ebba` | chrome | 0 | 1 | - |
| `93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9` | chrome | 0 | 1 | - |
| `9f935cce-0a9f-435f-8007-817732bfc0a5` | chrome | 0 | 1 | - |
| `c1fa57f3-c3db-4596-8f09-020701085416` | chrome | 0 | 1 | - |
| `f0b971a1-6831-4b9b-a50e-22a6e47f45ba` | chrome | 0 | 1 | - |

### run_3

**Total corrections:** 9

| Task ID | App | Original Reward | New Reward | Notes |
|---------|-----|-----------------|------------|-------|
| `2888b4e6-5b47-4b57-8bf5-c73827890774` | chrome | 0 | 1 | - |
| `3720f614-37fd-4d04-8a6b-76f54f8c222d` | chrome | 0 | 1 | - |
| `6c4c23a1-42a4-43cc-9db1-2f86ff3738cc` | chrome | 0 | 1 | - |
| `82bc8d6a-36eb-4d2d-8801-ef714fb1e55a` | chrome | 0 | 1 | - |
| `93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9` | chrome | 0 | 1 | - |
| `9f935cce-0a9f-435f-8007-817732bfc0a5` | chrome | 0 | 1 | - |
| `fc6d8143-9452-4171-9459-7f515143419a` | chrome | 0 | 1 | - |
| `d06f0d4d-2cd5-4ede-8de9-598629438c6e` | vlc | 0 | 1 | The agent downloaded a dark-theme skin from the internet and successfully set it as the VLC media player's default. The volume slider is black and overall the design is in a dark color scheme. |
| `e2dd0213-26db-4349-abe5-d5667bfd725c` | gimp | 0 | 1 | Task asks agent to move a centered text box to the left without specifying how much the offset should be. Agent moves the text box but not as much as where the evaluation function expects it to be. |
---

## All Corrected Tasks

The following tasks were corrected in at least one model/trial.

**Total unique tasks corrected:** 20

```
06fe7178-4491-4589-810f-2e2bc9502122
0d8b7de3-e8de-4d86-b9fd-dd2dce58a217
121ba48f-9e17-48ce-9bc6-a4fb17a7ebba
2888b4e6-5b47-4b57-8bf5-c73827890774
3720f614-37fd-4d04-8a6b-76f54f8c222d
550ce7e7-747b-495f-b122-acdc4d0b8e54
59155008-fe71-45ec-8a8f-dc35497b6aa8
6c4c23a1-42a4-43cc-9db1-2f86ff3738cc
82bc8d6a-36eb-4d2d-8801-ef714fb1e55a
841b50aa-df53-47bd-a73a-22d3a9f73160
93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9
9f935cce-0a9f-435f-8007-817732bfc0a5
b4f95342-463e-4179-8c3f-193cd7241fb2
b7895e80-f4d1-4648-bee0-4eb45a6f1fa8
c1fa57f3-c3db-4596-8f09-020701085416
d06f0d4d-2cd5-4ede-8de9-598629438c6e
da46d875-6b82-4681-9284-653b0c7ae241
f0b971a1-6831-4b9b-a50e-22a6e47f45ba
fc6d8143-9452-4171-9459-7f515143419a
e2dd0213-26db-4349-abe5-d5667bfd725c
```