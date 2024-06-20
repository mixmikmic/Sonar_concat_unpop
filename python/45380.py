# ## What does Reddit think about the new MBP?
# ### Is it da bomb, or did it just bomb?
# [**Run this notebook in binder**](http://mybinder.org:/repo/knowsuchagency/mpb-sentiment-analysis-example)
# 
# My friends and I have been talking today about Apple's announcement of their new Macbook Pro line.
# I personally own an Apple TV, Ipad mini, Macbook Pro, and an iPhone. I was definitely looking forward
# to see what Apple was going to come out with the new Macbook Pro line. My thoughts on the announcement aside,
# it seemed to me like the overwhelming majority of users on Reddit [didn't come away very impressed](https://www.reddit.com/r/apple/comments/59plnp/lets_talk_about_those_prices/) with the announcements Apple made in regards to the new Macbook Pro. I thought this would be a good opportunity to play with Reddit's API and to try out some rudimentary sentiment analysis.
# 

# ## Tech
# 
# To talk to Reddit, I'm using the aptly named, [**PRAW**](https://praw.readthedocs.io/en/stable/index.html) library. PRAW stands for the Python Reddit Api Wrapper. Nice.
# 
# Now for the sentiment analysis, I'm going to use the [**TextBlob**](https://textblob.readthedocs.io/en/dev/) library. TextBlob is a library that provides an easy to use interface for a lot of common natural language processing tasks.
# 

# ## Get the data
# 
# To begin, we instantiate praw's Reddit class with an appropriate user agent. Be careful how you define your user agent. Doing so incorrectly can get you banned according to Reddit. Fortunately, writing a user agent isn't hard at all. You can find the official documentation on how to do so on the [Reddit API wiki page](https://github.com/reddit/reddit/wiki/API). 
# 
# As of this writing, the format should look something like this (taken from the aformentioned wiki):
# 
#     <platform>:<app ID>:<version string> (by /u/<reddit username>)
# 
# Example: 
# 
#     User-Agent: android:com.example.myredditapp:v1.2.3 (by /u/kemitche)
# 

import praw

# I saved my user agent string in a local file
# since one should use their own
with open('./.user_agent') as f:
    user_agent = f.read()

# instantiate Reddit connection class
r = praw.Reddit(user_agent=user_agent)

# let's get the current top 10 submissions
# since praw interacts with reddit lazily, we wrap the method
# call in a list
submissions = list(r.get_subreddit('apple').get_hot(limit=10))
[str(s) for s in submissions]


# Cool, so we have some data. From looking at the actual subreddit in a browser, we can see the two top submissions are official threads so we'll just skip over them.
# 
# <a href="http://imgur.com/8MJXgdg"><img src="http://i.imgur.com/8MJXgdg.png" title="source: imgur.com" /></a>
# 

submissions = submissions[2:]
[str(s) for s in submissions]


# ### Submission 1
# 
# We'll start by looking at the comments in the first submission about Apple's new pricing. Can you guess what people think?!
# 
# In praw, every Submission has a comments attribute which is iterable. This attribute isn't homogeneous. That is, some of the items will be Comment objects, and there may be a MoreComments class in there as well, so we'll need to handle that.
# 

# grab the first submission
submission = submissions[0]

# the actual text is in the body
# attribute of a Comment
def get_comments(submission, n=20):
    """
    Return a list of comments from a submission.
    
    We can't just use submission.comments, because some
    of those comments may be MoreComments classes and those don't
    have bodies.
    
    n is the number of comments we want to limit ourselves to
    from the submission.
    """
    count = 0
    def barf_comments(iterable=submission.comments):
        """
        This generator barfs out comments.
        
        Some comments seem not to have bodies, so we skip those.
        """
        nonlocal count
        for c in iterable:
            if hasattr(c, 'body') and count < n:
                count += 1
                yield c.body
            elif hasattr(c, '__iter__'):
                # handle MoreComments classes
                yield from barf_comments(c)
            else:
                # c was a Comment and did not have a body
                continue
    return barf_comments()
                
comments = list(get_comments(submission))
list(comments)


# ## Sentiment analysis!
# 
# So now we have the first twenty comments of the first submission.
# 
# We'll combine them into one piece of text and determine the overall sentiment from them.
# 
# According to the TextBlob docs, this is how to use their sentiment analysis api and how to interpret it
# 
# 
# ### The sentiment property returns a namedtuple of the form Sentiment(polarity, subjectivity). The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
# 
# ```
# >>> testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
# >>> testimonial.sentiment
# Sentiment(polarity=0.39166666666666666, subjectivity=0.4357142857142857)
# >>> testimonial.sentiment.polarity
# 0.39166666666666666
# ```
# 

from textblob import TextBlob

comment_blob = TextBlob(''.join(comments))


# Huh, things aren't looking good so far. Well, let's look at more data.
# 

more_comments = []


for submission in submissions:
    more_comments.extend(get_comments(submission, 200))


len(more_comments)


# I figure there are many more than 404 comments for those submissions. I suspect we only get that many because praw [tries to do the right thing](https://praw.readthedocs.io/en/stable/pages/faq.html) and follow Reddit's guidelines for the amount of requests one can make within a given time limit. We're not going to worry about that now. 404 comments is good enough since we're only tinkering :)
# 

bigger_blob = TextBlob(''.join(more_comments))

# The first time I ran this method, it failed
# because I hadn't read TextBlob's docs closely
# and downloaded the corpus of text in needed.
# python -m textblob.download_corpora

print(len(bigger_blob.words))


# Let's see what some of the most common words are
# 

from collections import Counter

counter = Counter(bigger_blob.words)

# the most common words are pretty mundane common parts of speech, so we'll skip the first few
counter.most_common()[60:100]


# ### Finally, let's see what the over all sentiment analysis is
# 

bigger_blob.sentiment


# We see that the overall sentiment is much more positive when we include a larger body of comments.
# 

# ## In conclusion
# 
# We've hopefully learned a little more about communicating with Reddit using Python and doing some simple sentiment analysis on the content there. This wasn't meant to be a very scientific excercise, but I thought it was a fun way to play around with [PRAW](https://praw.readthedocs.io/en/stable/index.html) and [TextBlob](https://textblob.readthedocs.io/en/dev/index.html). Both libraries are really powerful and simple to use and I can definitely see myself taking advantage of them a lot more in the future. 
# 




