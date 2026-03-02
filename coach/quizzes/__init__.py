"""Quiz questions for coach/quizzes/__init__.py"""

def load_all_questions():
    from coach.quizzes import (
        ml_design_prep, visual_search, street_view, video_search,
        harmful_content, video_rec, event_rec, ad_click,
        similar_listing, news_feed, pymk,
        rnn_fundamentals, transformers,
    )
    all_q = []
    for mod in [ml_design_prep, visual_search, street_view, video_search,
                harmful_content, video_rec, event_rec, ad_click,
                similar_listing, news_feed, pymk,
                rnn_fundamentals, transformers]:
        all_q.extend(mod.QUESTIONS)
    return all_q
