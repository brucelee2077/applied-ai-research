"""Quiz questions for Module 11: People You May Know (PYMK)"""

QUESTIONS = [
    {
        "concept_id": "pymk_graph_features",
        "module": "11-people-you-may-know",
        "question": "What is the single most predictive feature for 'People You May Know' recommendations?",
        "choices": [
            "A. Similar job title",
            "B. Mutual friends count — two people with 10+ mutual friends are very likely to know each other in real life",
            "C. Same city",
            "D. Similar profile photo"
        ],
        "correct": "B",
        "hint": "Think about Triadic Closure theory: if A knows B and B knows C, what is likely?",
        "explanation": "Mutual friends is the dominant signal for PYMK. Triadic Closure (a foundational social network theory): if two people share many mutual friends, they are very likely to know each other. Empirically, people with 10+ mutual friends accept connection requests at 60%+ rate. This far outperforms demographic similarity, same school, or co-workers alone.",
        "difficulty": 2,
        "tags": ["graph_features", "mutual_friends", "triadic_closure"]
    },
    {
        "concept_id": "pymk_graph_embeddings",
        "module": "11-people-you-may-know",
        "question": "For a graph with 3B users, exact computation of mutual friends for all pairs is infeasible. What is the scalable approach?",
        "choices": [
            "A. Only compute for users with > 500 friends",
            "B. Graph embeddings (GraphSAGE, Node2Vec): each user is embedded based on their neighborhood structure. Similar neighborhoods → nearby embeddings. Use ANN on embeddings instead of exact mutual friend computation.",
            "C. Sample 1% of pairs randomly",
            "D. Use only profile text similarity"
        ],
        "correct": "B",
        "hint": "You can't enumerate all 3B × 3B pairs. What representation allows you to find similar nodes efficiently?",
        "explanation": "Graph embeddings compress each node's neighborhood structure into a dense vector. Users with similar social neighborhoods (similar friend groups, interaction patterns) get similar embeddings. Graph neural networks (GraphSAGE) aggregate neighborhood features iteratively. ANN search over embeddings replaces exact pair-wise mutual-friend computation, making PYMK scalable to billions of users.",
        "difficulty": 4,
        "tags": ["graph_embeddings", "graphsage", "scalability"]
    },
    {
        "concept_id": "pymk_address_book",
        "module": "11-people-you-may-know",
        "question": "Facebook's PYMK uses phone contact book uploads. What is the privacy concern and how is it addressed?",
        "choices": [
            "A. No privacy concern — users consent",
            "B. Consent ambiguity: users upload their contacts but the contacts haven't consented to having their phone numbers used to discover social connections. Privacy-preserving hashing + on-device processing mitigates this.",
            "C. The data is too large to store",
            "D. Contact books are not reliable"
        ],
        "correct": "B",
        "hint": "Who consents when a user uploads their contacts?",
        "explanation": "Contact book PYMK creates a third-party privacy issue: person A uploads their contacts including person B's phone number. Person B never consented to their phone number being used for social graph inference. Mitigation: (1) Hash phone numbers before upload (so raw numbers aren't stored), (2) Only use for mutual-contact signals (not direct lookup), (3) Allow users to opt out of contact-based discovery. This was a major issue in Facebook's 2018 privacy investigations.",
        "difficulty": 3,
        "tags": ["privacy", "contact_book", "consent"]
    },
    {
        "concept_id": "pymk_precision_over_recall",
        "module": "11-people-you-may-know",
        "question": "For PYMK, should you optimize for high precision or high recall? Why?",
        "choices": [
            "A. High recall — suggest as many potential connections as possible",
            "B. High precision — a false positive (suggesting a stranger) is socially awkward and users tune out PYMK entirely. A false negative (missing a real acquaintance) is much less harmful.",
            "C. Equal precision and recall (F1)",
            "D. Recall doesn't matter for social networks"
        ],
        "correct": "B",
        "hint": "What happens to user trust if PYMK suggests people the user has never met?",
        "explanation": "PYMK precision vs. recall tradeoff: a false positive (suggesting a stranger) erodes user trust in the feature — users who see irrelevant suggestions stop engaging with PYMK. A false negative (not suggesting someone you know) is acceptable because that person might appear later as you add more mutual connections. This asymmetry means PYMK should be conservative — only suggest when confidence is high.",
        "difficulty": 3,
        "tags": ["precision_recall", "user_trust", "tradeoffs"]
    },
    {
        "concept_id": "pymk_temporal_signals",
        "module": "11-people-you-may-know",
        "question": "You attended a conference and exchanged business cards with 20 people. How should PYMK surface these connections?",
        "choices": [
            "A. Only surface them if they already sent a friend request",
            "B. Time-aware cooccurrence: users at the same location/event at the same time are a strong PYMK signal. Temporal proximity + spatial proximity → suggest connections in the days following the event",
            "C. Show them only if they have 5+ mutual friends",
            "D. Wait until they visit each other's profiles"
        ],
        "correct": "B",
        "hint": "You just met someone in person — what signals does your phone know about that encounter?",
        "explanation": "Physical co-presence signals: location check-ins, GPS proximity events (with privacy-preserving aggregation), same event hashtags, tagged in same photo, same WiFi network. Temporal decay: these signals are strongest immediately after the encounter (people connect on LinkedIn/Facebook within 24-48h of meeting). Event-aware PYMK surfaces these connections in the peak window of recall.",
        "difficulty": 3,
        "tags": ["temporal_signals", "co_presence", "event_signals"]
    },
    {
        "concept_id": "pymk_homophily_bias",
        "module": "11-people-you-may-know",
        "question": "PYMK trained on historical connection data will exhibit homophily bias. What is this and why is it a concern?",
        "choices": [
            "A. The model suggests only younger users",
            "B. Homophily: people connect with those similar to them (same race, class, education). PYMK trained on historical data amplifies these existing social divisions, reducing cross-demographic connection opportunities.",
            "C. The model is too slow",
            "D. Users connect with too many strangers"
        ],
        "correct": "B",
        "hint": "Social networks naturally cluster by demographics. What happens if your recommendation model reinforces these clusters?",
        "explanation": "Homophily bias: people naturally tend to connect with those similar to themselves (same race 86%, same education level 82% of connections). PYMK trained on historical data learns to recommend same-demographic connections, further deepening social segregation. This has been documented as a concern in research. Mitigation approaches: diversity-aware ranking that occasionally promotes cross-demographic recommendations, though this requires explicit policy decisions about what 'fairness' means.",
        "difficulty": 4,
        "tags": ["bias", "homophily", "fairness", "social_network"]
    },
    {
        "concept_id": "pymk_bilateral_consent",
        "module": "11-people-you-may-know",
        "question": "Should PYMK show person A in person B's suggestions even if A has blocked B? What about mutual blocking?",
        "choices": [
            "A. Block status is irrelevant to PYMK",
            "B. Block status is a hard filter: if either A blocked B OR B blocked A, they should NEVER appear in each other's PYMK. This is a critical safety feature, not a preference.",
            "C. Only hide from suggestions if both users blocked each other",
            "D. Show with a warning that the person may not want contact"
        ],
        "correct": "B",
        "hint": "Someone blocks another person often for safety reasons. What happens if they still appear in suggestions?",
        "explanation": "Block filtering is a hard safety requirement, not a soft preference. Reasons for blocking include harassment, domestic abuse, stalking. If a blocked person appears in PYMK suggestions, it reveals that person is on the platform and potentially exposes location/activity signals. This is a hard filter applied BEFORE any ML ranking — it is never overridden by the recommendation model. Also includes shadow ban and deactivated accounts.",
        "difficulty": 2,
        "tags": ["safety", "block_filter", "hard_constraints"]
    },
    {
        "concept_id": "pymk_multi_hop",
        "module": "11-people-you-may-know",
        "question": "2-hop connections (friends-of-friends) are strong PYMK signals. Why are 3-hop connections generally less useful?",
        "choices": [
            "A. 3-hop connections are too slow to compute",
            "B. Signal-to-noise degrades exponentially with hop distance: 2-hop connections are often actual acquaintances; 3-hop connections are mostly strangers on large networks. Also, with billions of nodes, 3-hop sets are astronomically large.",
            "C. Graph databases can't handle 3-hop queries",
            "D. 3-hop connections are equally good"
        ],
        "correct": "B",
        "hint": "In a network of 3B users, how many 3-hop connections does the average person have?",
        "explanation": "Hop distance and precision tradeoff: 2-hop (friends-of-friends) with high mutual friend count → very high PYMK precision. 3-hop: on average social networks, everyone is within 3 hops of everyone else ('six degrees of separation'). The 3-hop candidate set is essentially the entire network, providing no useful signal. Additionally, the probability of actually knowing a 3-hop connection is < 1% on large networks.",
        "difficulty": 3,
        "tags": ["graph_traversal", "hop_distance", "signal_quality"]
    },
    {
        "concept_id": "pymk_cold_start_new_user",
        "module": "11-people-you-may-know",
        "question": "A brand new user with 0 friends joins Facebook. What PYMK signals are available?",
        "choices": [
            "A. None — wait until they add friends",
            "B. Onboarding signals: email contacts, phone contacts, employer/school from profile, device location for local network, mutual connections with invited users",
            "C. Show globally popular users",
            "D. Show users from the same country"
        ],
        "correct": "B",
        "hint": "You have no graph data, but you have other data the user provided during signup.",
        "explanation": "New user PYMK bootstrap: (1) Email invitation path: they joined via an invite from person X → suggest X's connections, (2) Contact book upload at onboarding → immediate mutual-contact graph, (3) Profile data: school (class of 2015) → suggest classmates already on platform, employer → suggest colleagues, (4) Device location: suggest users physically nearby. These bootstrap signals generate first 5-10 connections, after which the graph-based model takes over.",
        "difficulty": 3,
        "tags": ["cold_start", "new_user", "bootstrap"]
    },
    {
        "concept_id": "pymk_connection_quality",
        "module": "11-people-you-may-know",
        "question": "After a PYMK connection is made, how do you evaluate if it was a 'good' suggestion?",
        "choices": [
            "A. Connection was made = success",
            "B. Long-term quality metrics: subsequent interactions (messages, likes, comments between the two users after connecting), not just connection acceptance rate — a connected but never-interacted pair is a low-quality suggestion",
            "C. Profile completeness of the suggested user",
            "D. Number of mutual friends added after connection"
        ],
        "correct": "B",
        "hint": "What's the difference between accepting a connection and actually caring about that connection?",
        "explanation": "PYMK quality measurement: connection acceptance rate is a short-term metric easily gamed by suggesting popular users. Long-term quality: do the two users actually interact after connecting (message, like, comment)? Post-connection interaction rate within 30/90 days is a much stronger signal that the PYMK suggestion was genuinely valuable. This feedback loop improves the training labels for the PYMK model over time.",
        "difficulty": 3,
        "tags": ["evaluation", "long_term_quality", "engagement"]
    },
]
