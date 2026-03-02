"""Quiz questions for Module 07: Event Recommendation"""

QUESTIONS = [
    {
        "concept_id": "erec_temporal_features",
        "module": "07-event-recommendation",
        "question": "Why are temporal features especially critical for event recommendation compared to content recommendation?",
        "choices": [
            "A. Events are harder to describe",
            "B. Events are time-bound — an event recommendation after it has occurred is worthless. Proximity to event date, user lead time preferences, and urgency signals are first-class features.",
            "C. Users watch events on YouTube",
            "D. Events are rarer than videos"
        ],
        "correct": "B",
        "hint": "What happens if you recommend a concert that happened yesterday?",
        "explanation": "Events have hard temporal constraints: recommending an event 1 hour before it starts is useful for spontaneous users but useless for planners who need 2 weeks. Key temporal features: days_until_event, user_typical_lead_time (how far in advance does this user usually book?), day-of-week patterns, seasonality. Stale recommendations are a severe UX failure unique to events.",
        "difficulty": 2,
        "tags": ["temporal_features", "event_specific"]
    },
    {
        "concept_id": "erec_location_context",
        "module": "07-event-recommendation",
        "question": "A user in San Francisco searches for 'jazz concert'. What location signals should the model use?",
        "choices": [
            "A. Only the user's registered home city",
            "B. A hierarchy: current GPS location, home city, past venues visited, travel radius preference — with geospatial distance as a hard filter before ML ranking",
            "C. Show all jazz concerts globally sorted by popularity",
            "D. Only events within 1 mile"
        ],
        "correct": "B",
        "hint": "Users attend events in their city but also travel. How do you balance this?",
        "explanation": "Location-aware recommendation: (1) Hard geospatial filter (only show events within user's implicit travel radius — learned from past behavior), (2) Distance as a feature (closer = higher score, but not absolute), (3) Context-awareness: user is currently in NYC (traveling) → show NYC events even if home is SF, (4) Travel radius preference: inferred from booking history (some users always go local, some travel 50+ miles for events).",
        "difficulty": 3,
        "tags": ["geospatial", "location_context", "features"]
    },
    {
        "concept_id": "erec_cold_start_event",
        "module": "07-event-recommendation",
        "question": "A brand new artist announces their first concert with no engagement history. How do you recommend this event?",
        "choices": [
            "A. Don't recommend it until it has reviews",
            "B. Use content-based signals: artist genre similarity to known artists, venue popularity, ticket price tier, similar-artist fan overlap from music streaming data",
            "C. Show it to all users equally",
            "D. Wait for early ticket sales to use as a signal"
        ],
        "correct": "B",
        "hint": "No engagement history → rely on what you DO know about the event.",
        "explanation": "Event cold start: leverage (1) Artist graph — if users who like Artist X like Artist Y, and Y is similar to Z (new artist), surface to X fans, (2) Venue history — the venue's past events attracted certain demographics, (3) Genre/category signals, (4) Ticket pricing as a proxy for expected audience size, (5) Cross-platform signals (Spotify followers, social media following for the artist).",
        "difficulty": 3,
        "tags": ["cold_start", "content_based", "cross_platform"]
    },
    {
        "concept_id": "erec_group_dynamics",
        "module": "07-event-recommendation",
        "question": "How do you recommend events to a group of friends planning to attend together?",
        "choices": [
            "A. Recommend to the most active user and ignore others",
            "B. Aggregate group preferences: compute intersection of individual preferences (everyone must be interested) or weighted average, with veto detection (if one member strongly dislikes a category, exclude it)",
            "C. Recommend the most popular events",
            "D. Groups cannot use recommendation systems"
        ],
        "correct": "B",
        "hint": "A horror movie fan and a family with kids have incompatible preferences. How do you satisfy both?",
        "explanation": "Group recommendation is a distinct problem: simple averaging fails when preferences are polarized. Strategies: (1) Least misery: maximize the minimum satisfaction across group members (a weak veto), (2) Maximum fairness: rotate who gets 'their pick', (3) Intersection approach: only recommend events all members would individually rate positive. The correct choice depends on group dynamics (friends = some compromise ok; family = strong veto power for kids).",
        "difficulty": 4,
        "tags": ["group_recommendation", "preference_aggregation"]
    },
    {
        "concept_id": "erec_scarcity_signal",
        "module": "07-event-recommendation",
        "question": "A concert has only 50 tickets remaining. How should this scarcity signal be used?",
        "choices": [
            "A. Remove it from recommendations (almost sold out)",
            "B. Use scarcity as a re-ranking boost for users already interested, paired with urgency notification. Also use it as a signal to increase recommendation frequency for high-intent users.",
            "C. Show it to all users regardless of interest",
            "D. Scarcity signals are not useful for ML"
        ],
        "correct": "B",
        "hint": "Low availability is relevant to some users (interested → buy now) but not all (low interest → irrelevant).",
        "explanation": "Scarcity interacts with user interest: high-interest × low-availability = high urgency to show. Low-interest × low-availability = irrelevant. Features: available_tickets, ticket_velocity (how fast are tickets selling), time_until_sellout_estimate. For interested users: surface higher in ranking + trigger urgency notification. For uninterested users: don't show just because it's scarce.",
        "difficulty": 3,
        "tags": ["scarcity", "urgency", "feature_engineering"]
    },
    {
        "concept_id": "erec_session_context",
        "module": "07-event-recommendation",
        "question": "A user spends 20 minutes browsing EDM events but hasn't bought yet. What does this session behavior signal?",
        "choices": [
            "A. The user is not interested in EDM",
            "B. High intent for EDM events — the session dwell time and browsing pattern indicate purchase consideration. Use this real-time session signal to boost EDM events in ranking.",
            "C. The user is just casually browsing",
            "D. Session data should not be used in recommendation"
        ],
        "correct": "B",
        "hint": "20 minutes of browsing a category is a stronger signal than a casual scroll.",
        "explanation": "Session-level intent modeling: dwell time, scroll depth, repeat views of the same event, price page visits all signal purchase intent. Real-time session features should override or strongly modulate long-term user profile signals. A user who normally books classical music but spends 20 min on EDM is currently in 'EDM mode' — serve that intent now.",
        "difficulty": 3,
        "tags": ["session_modeling", "real_time", "intent"]
    },
    {
        "concept_id": "erec_serendipity",
        "module": "07-event-recommendation",
        "question": "How do you balance recommending familiar events (safe bets) vs. introducing users to new genres/venues?",
        "choices": [
            "A. Always recommend familiar events",
            "B. Serendipity-aware ranking: intentionally include a fraction (10-20%) of 'stretch' recommendations — outside normal patterns but not completely random. Measure long-term satisfaction lift from serendipitous recommendations.",
            "C. Never recommend unfamiliar events",
            "D. Let users control novelty with a filter"
        ],
        "correct": "B",
        "hint": "Users often say they want familiar recommendations but are delighted when they discover something new.",
        "explanation": "The serendipity problem: pure relevance maximization creates boring homogenous feeds. Users don't know what they don't know — they can't ask for genres they've never experienced. Solution: inject a controlled fraction of exploratory recommendations, measured by: (1) do users attend them? (2) do they rate them higher than expected? (3) does it expand their future interest graph? This is a classic explore-exploit problem.",
        "difficulty": 3,
        "tags": ["serendipity", "diversity", "exploration"]
    },
    {
        "concept_id": "erec_cancellation_handling",
        "module": "07-event-recommendation",
        "question": "A recommended event gets cancelled after the user clicks 'interested'. How does this affect the ML system?",
        "choices": [
            "A. No impact — just remove from future recommendations",
            "B. Label correction needed: the positive signal (click/interest) was generated before cancellation. Remove these examples from training or mark as cancelled to avoid training the model to recommend cancelled events' features.",
            "C. Use the click as a positive signal anyway",
            "D. Block the event organizer"
        ],
        "correct": "B",
        "hint": "Your model learns: 'events like this get clicks'. But those clicks happened before cancellation was known.",
        "explanation": "Cancelled events create spurious positive labels. If you train on clicks for events that later cancel, the model learns to recommend 'events that cancel' (which may share systematic features like certain organizers, venue types, or price points). Solutions: (1) retroactively nullify clicks on cancelled events, (2) add 'cancellation_rate' as a feature per organizer/venue as a reliability signal.",
        "difficulty": 4,
        "tags": ["label_noise", "data_quality", "temporal"]
    },
    {
        "concept_id": "erec_social_signals",
        "module": "07-event-recommendation",
        "question": "Your friend bought tickets to a concert. How should the recommendation system use this social signal?",
        "choices": [
            "A. Ignore it — recommendation is individual",
            "B. Strong positive signal: social proof from trusted connections is a high-precision indicator of interest. Surface the event with social context ('3 friends are going').",
            "C. Only use if the user explicitly asks for social recommendations",
            "D. Show it only if the friend agrees to share"
        ],
        "correct": "B",
        "hint": "Word of mouth is the strongest form of recommendation.",
        "explanation": "Social graph signals are among the highest-precision recommendation signals: if a close friend attends an event, the probability of your interest is significantly above base rate. Implementation: social proof display ('3 friends going'), friend-going signal as a feature in ranking model, social graph filtering (only events at least one friend attended or expressed interest in). Privacy: only use explicitly shared social data.",
        "difficulty": 2,
        "tags": ["social_signals", "social_proof", "collaborative_filtering"]
    },
    {
        "concept_id": "erec_supply_demand",
        "module": "07-event-recommendation",
        "question": "There are 1000x more users than event spots for a sold-out stadium concert. How should the recommendation system allocate 'recommendation attention'?",
        "choices": [
            "A. Show it to all users equally — first come first served",
            "B. Recommend primarily to highest-intent users (predicted purchase probability × engagement), and suppress it for casual browsers to avoid building demand that can't be fulfilled",
            "C. Only show it to VIP users",
            "D. Don't show sold-out events at all"
        ],
        "correct": "B",
        "hint": "What happens if you recommend a sold-out event to millions of users who then can't buy?",
        "explanation": "Supply-constrained recommendation: showing a sold-out event to low-intent users wastes their attention and creates frustration when they can't buy. Optimize for: (1) Waitlist conversion (show to high-intent users who might want a waitlist spot), (2) Similar event discovery ('this is sold out, but here are similar events with tickets'), (3) Demand forecasting: use early interest signals to predict which events need early promotion.",
        "difficulty": 4,
        "tags": ["supply_constraint", "demand_forecasting", "optimization"]
    },
]
