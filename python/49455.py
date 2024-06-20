# # Interactive Data Visualization with Bokeh
# 
# [Bokeh](http://bokeh.pydata.org/en/latest/) is an interactive Python library for visualizations that targets modern web browsers for presentation. Its goal is to provide elegant, concise construction of novel graphics in the style of D3.js, and to extend this capability with high-performance interactivity over very large or streaming datasets. Bokeh can help anyone who would like to quickly and easily create interactive plots, dashboards, and data applications.
# 
#  - To get started using Bokeh to make your visualizations, see the [User Guide](http://bokeh.pydata.org/en/latest/docs/user_guide.html#userguide).
#  - To see examples of how you might use Bokeh with your own data, check out the [Gallery](http://bokeh.pydata.org/en/latest/docs/gallery.html#gallery).
#  - A complete API reference of Bokeh is at [Reference Guide](http://bokeh.pydata.org/en/latest/docs/reference.html#refguide).
# 
# The following notebook is intended to illustrate some of Bokeh's interactive utilities and is based on a [post](https://demo.bokehplots.com/apps/gapminder) by software engineer and Bokeh developer [Sarah Bird](https://twitter.com/birdsarah).
# 
# 
# ## Recreating Gapminder's "The Health and Wealth of Nations" 
# 
# Gapminder started as a spin-off from Professor Hans Rosling’s teaching at the Karolinska Institute in Stockholm. Having encountered broad ignorance about the rapid health improvement in Asia, he wanted to measure that lack of awareness among students and professors. He presented the surprising results from his so-called “Chimpanzee Test” in [his first TED-talk](https://www.ted.com/talks/hans_rosling_shows_the_best_stats_you_ve_ever_seen) in 2006.
# 
# [![The Best Stats You've Never Seen](http://img.youtube.com/vi/hVimVzgtD6w/0.jpg)](http://www.youtube.com/watch?v=hVimVzgtD6w "The best stats you've ever seen | Hans Rosling")
# 
# Rosling's interactive ["Health and Wealth of Nations" visualization](http://www.gapminder.org/world) has since become an iconic  illustration of how our assumptions about ‘first world’ and ‘third world’ countries can betray us. Mike Bostock has [recreated the visualization using D3.js](https://bost.ocks.org/mike/nations/), and in this lab, we will see that it is also possible to use Bokeh to recreate the interactive visualization in Python.
# 
# 
# ### About Bokeh Widgets
# Widgets are interactive controls that can be added to Bokeh applications to provide a front end user interface to a visualization. They can drive new computations, update plots, and connect to other programmatic functionality. When used with the [Bokeh server](http://bokeh.pydata.org/en/latest/docs/user_guide/server.html), widgets can run arbitrary Python code, enabling complex applications. Widgets can also be used without the Bokeh server in standalone HTML documents through the browser’s Javascript runtime.
# 
# To use widgets, you must add them to your document and define their functionality. Widgets can be added directly to the document root or nested inside a layout. There are two ways to program a widget’s functionality:
# 
#  - Use the CustomJS callback (see [CustomJS for Widgets](http://bokeh.pydata.org/en/0.12.0/docs/user_guide/interaction.html#userguide-interaction-actions-widget-callbacks). This will work in standalone HTML documents.
#  - Use `bokeh serve` to start the Bokeh server and set up event handlers with `.on_change` (or for some widgets, `.on_click`).
#  
# ### Imports
# 

# Science Stack 
import numpy as np
import pandas as pd

# Bokeh Essentials 
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, ColumnDataSource

# Layouts 
from bokeh.layouts import layout
from bokeh.layouts import widgetbox

# Figure interaction layer
from bokeh.io import show
from bokeh.io import output_notebook 

# Data models for visualization 
from bokeh.models import Text
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import HoverTool
from bokeh.models import LinearAxis
from bokeh.models import ColumnDataSource
from bokeh.models import SingleIntervalTicker

# Palettes and colors
from bokeh.palettes import brewer
from bokeh.palettes import Spectral6


# To display Bokeh plots inline in a Jupyter notebook, use the `output_notebook()` function from bokeh.io. When `show()` is called, the plot will be displayed inline in the next notebook output cell. To save your Bokeh plots, you can use the `output_file()` function instead (or in addition). The `output_file()` function will write an HTML file to disk that can be opened in a browser. 
# 

# Load Bokeh for visualization
output_notebook()


# ### Get the data
# 
# Some of Bokeh examples rely on sample data that is not included in the Bokeh GitHub repository or released packages, due to their size. Once Bokeh is installed, the sample data can be obtained by executing the command in the next cell. The location that the sample data is stored can be configured. By default, data is downloaded and stored to a directory $HOME/.bokeh/data. (The directory is created if it does not already exist.) 
# 

import bokeh.sampledata
bokeh.sampledata.download()


# ### Prepare the data    
#  
# In order to create an interactive plot in Bokeh, we need to animate snapshots of the data over time from 1964 to 2013. In order to do this, we can think of each year as a separate static plot. We can then use a JavaScript `Callback` to change the data source that is driving the plot.    
# 
# #### JavaScript Callbacks
# 
# Bokeh exposes various [callbacks](http://bokeh.pydata.org/en/latest/docs/user_guide/interaction/callbacks.html#userguide-interaction-callbacks), which can be specified from Python, that trigger actions inside the browser’s JavaScript runtime. This kind of JavaScript callback can be used to add interesting interactions to Bokeh documents without the need to use a Bokeh server (but can also be used in conjuction with a Bokeh server). Custom callbacks can be set using a [`CustomJS` object](http://bokeh.pydata.org/en/latest/docs/user_guide/interaction/callbacks.html#customjs-for-widgets) and passing it as the callback argument to a `Widget` object.
# 
# As the data we will be using today is not too big, we can pass all the datasets to the JavaScript at once and switch between them on the client side using a slider widget.    
# 
# This means that we need to put all of the datasets together build a single data source for each year. First we will load each of the datasets with the `process_data()` function and do a bit of clean up:
# 

def process_data():
    
    # Import the gap minder data sets
    from bokeh.sampledata.gapminder import fertility, life_expectancy, population, regions
    
    # The columns are currently string values for each year, 
    # make them ints for data processing and visualization.
    columns = list(fertility.columns)
    years = list(range(int(columns[0]), int(columns[-1])))
    rename_dict = dict(zip(columns, years))
    
    # Apply the integer year columna names to the data sets. 
    fertility = fertility.rename(columns=rename_dict)
    life_expectancy = life_expectancy.rename(columns=rename_dict)
    population = population.rename(columns=rename_dict)
    regions = regions.rename(columns=rename_dict)

    # Turn population into bubble sizes. Use min_size and factor to tweak.
    scale_factor = 200
    population_size = np.sqrt(population / np.pi) / scale_factor
    min_size = 3
    population_size = population_size.where(population_size >= min_size).fillna(min_size)

    # Use pandas categories and categorize & color the regions
    regions.Group = regions.Group.astype('category')
    regions_list = list(regions.Group.cat.categories)

    def get_color(r):
        return Spectral6[regions_list.index(r.Group)]
    regions['region_color'] = regions.apply(get_color, axis=1)

    return fertility, life_expectancy, population_size, regions, years, regions_list


# Next we will add each of our sources to the `sources` dictionary, where each key is the name of the year (prefaced with an underscore) and each value is a dataframe with the aggregated values for that year.
# 
# _Note that we needed the prefixing as JavaScript objects cannot begin with a number._
# 

# Process the data and fetch the data frames and lists 
fertility_df, life_expectancy_df, population_df_size, regions_df, years, regions = process_data()

# Create a data source dictionary whose keys are prefixed years
# and whose values are ColumnDataSource objects that merge the 
# various per-year values from each data frame. 
sources = {}

# Quick helper variables 
region_color = regions_df['region_color']
region_color.name = 'region_color'

# Create a source for each year. 
for year in years:
    # Extract the fertility for each country for this year.
    fertility = fertility_df[year]
    fertility.name = 'fertility'
    
    # Extract life expectancy for each country for this year. 
    life = life_expectancy_df[year]
    life.name = 'life' 
    
    # Extract the normalized population size for each country for this year. 
    population = population_df_size[year]
    population.name = 'population' 
    
    # Create a dataframe from our extraction and add to our sources 
    new_df = pd.concat([fertility, life, population, region_color], axis=1)
    sources['_' + str(year)] = ColumnDataSource(new_df)


# You can see what's in the `sources` dictionary by running the cell below.
# 
# Later we will be able to pass this `sources` dictionary to the JavaScript Callback. In so doing, we will find that in our JavaScript we have objects named by year that refer to a corresponding `ColumnDataSource`.
# 

sources


# We can also create a corresponding `dictionary_of_sources` object, where the keys are integers and the values are the references to our ColumnDataSources from above: 
# 

dictionary_of_sources = dict(zip([x for x in years], ['_%s' % x for x in years]))


js_source_array = str(dictionary_of_sources).replace("'", "")
js_source_array


# Now we have an object that's storing all of our `ColumnDataSources`, so that we can look them up.
# 
# ### Build the plot
# 
# First we need to create a `Plot` object. We'll start with a basic frame, only specifying things like plot height, width, and ranges for the axes.
# 

xdr = Range1d(1, 9)
ydr = Range1d(20, 100)

plot = Plot(
    x_range=xdr,
    y_range=ydr,
    plot_width=800,
    plot_height=400,
    outline_line_color=None,
    toolbar_location=None, 
    min_border=20,
)


# In order to display the plot in the notebook use the `show()` function:
# 

# show(plot)


# ### Build the axes
# 
# Next we can make some stylistic modifications to the plot axes (e.g. by specifying the text font, size, and color, and by adding labels), to make the plot look more like the one in Hans Rosling's video.
# 

# Create a dictionary of our common setting. 
AXIS_FORMATS = dict(
    minor_tick_in=None,
    minor_tick_out=None,
    major_tick_in=None,
    major_label_text_font_size="10pt",
    major_label_text_font_style="normal",
    axis_label_text_font_size="10pt",

    axis_line_color='#AAAAAA',
    major_tick_line_color='#AAAAAA',
    major_label_text_color='#666666',

    major_tick_line_cap="round",
    axis_line_cap="round",
    axis_line_width=1,
    major_tick_line_width=1,
)


# Create two axis models for the x and y axes. 
xaxis = LinearAxis(
    ticker=SingleIntervalTicker(interval=1), 
    axis_label="Children per woman (total fertility)", 
    **AXIS_FORMATS
)

yaxis = LinearAxis(
    ticker=SingleIntervalTicker(interval=20), 
    axis_label="Life expectancy at birth (years)", 
    **AXIS_FORMATS
)   

# Add the axes to the plot in the specified positions.
plot.add_layout(xaxis, 'below')
plot.add_layout(yaxis, 'left')


# Go ahead and experiment with visualizing each step of the building process and changing various settings.
# 

# show(plot)


# ### Add the background year text
# 
# One of the features of Rosling's animation is that the year appears as the text background of the plot. We will add this feature to our plot first so it will be layered below all the other glyphs (will will be incrementally added, layer by layer, on top of each other until we are finished).
# 

# Create a data source for each of our years to display. 
text_source = ColumnDataSource({'year': ['%s' % years[0]]})

# Create a text object model and add to the figure. 
text = Text(x=2, y=35, text='year', text_font_size='150pt', text_color='#EEEEEE')
plot.add_glyph(text_source, text)


# show(plot)


# ### Add the bubbles and hover
# Next we will add the bubbles using Bokeh's [`Circle`](http://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.circle) glyph. We start from the first year of data, which is our source that drives the circles (the other sources will be used later).    
# 

# Select the source for the first year we have. 
renderer_source = sources['_%s' % years[0]]

# Create a circle glyph to generate points for the scatter plot. 
circle_glyph = Circle(
    x='fertility', y='life', size='population',
    fill_color='region_color', fill_alpha=0.8, 
    line_color='#7c7e71', line_width=0.5, line_alpha=0.5
)

# Connect the glyph generator to the data source and add to the plot
circle_renderer = plot.add_glyph(renderer_source, circle_glyph)


# In the above, `plot.add_glyph` returns the renderer, which we can then pass to the `HoverTool` so that hover only happens for the bubbles on the page and not other glyph elements:
# 

# Add the hover (only against the circle and not other plot elements)
tooltips = "@index"
plot.add_tools(HoverTool(tooltips=tooltips, renderers=[circle_renderer]))


# Test out different parameters for the `Circle` glyph and see how it changes the plot:
# 

# show(plot)


# ### Add the legend
# 
# Next we will manually build a legend for our plot by adding circles and texts to the upper-righthand portion:
# 

# Position of the legend 
text_x = 7
text_y = 95

# For each region, add a circle with the color and text. 
for i, region in enumerate(regions):
    plot.add_glyph(Text(x=text_x, y=text_y, text=[region], text_font_size='10pt', text_color='#666666'))
    plot.add_glyph(
        Circle(x=text_x - 0.1, y=text_y + 2, fill_color=Spectral6[i], size=10, line_color=None, fill_alpha=0.8)
    )
    
    # Move the y coordinate down a bit.
    text_y = text_y - 5


# show(plot)


# ### Add the slider and callback
# Next we add the slider widget and the JavaScript callback code, which changes the data of the `renderer_source` (powering the bubbles / circles) and the data of the `text_source` (powering our background text). After we've `set()` the data we need to `trigger()` a change. `slider`, `renderer_source`, `text_source` are all available because we add them as args to `Callback`.    
# 
# It is the combination of `sources = %s % (js_source_array)` in the JavaScript and `Callback(args=sources...)` that provides the ability to look-up, by year, the JavaScript version of our Python-made `ColumnDataSource`.
# 

# Add the slider
code = """
    var year = slider.get('value'),
        sources = %s,
        new_source_data = sources[year].get('data');
    renderer_source.set('data', new_source_data);
    text_source.set('data', {'year': [String(year)]});
""" % js_source_array

callback = CustomJS(args=sources, code=code)
slider = Slider(start=years[0], end=years[-1], value=1, step=1, title="Year", callback=callback)
callback.args["renderer_source"] = renderer_source
callback.args["slider"] = slider
callback.args["text_source"] = text_source


# show(widgetbox(slider))


# ### Putting all the pieces together
# 
# Last but not least, we put the chart and the slider together in a layout and display it inline in the notebook.
# 

show(layout([[plot], [slider]], sizing_mode='scale_width'))


# I hope that you'll use Bokeh to produce interactive visualizations for visual analysis:
# 
# ![The Visual Analytics Mantra](figures/visual_analytics_mantra.png)
# 
# ## Topic Model Visualization
# 
# In this section we'll take a look at visualizing a corpus by exploring clustering and dimensionality reduction techniques. Text analysis is certainly high dimensional visualization and this can be applied to other data sets as well. 
# 
# The first step is to load our documents from disk and vectorize them using Gensim. This content is a bit beyond the scope of the workshop for today, however I did want to provide code for reference, and I'm happy to go over it offline. 

import nltk 
import string
import pickle
import gensim
import random 

from operator import itemgetter
from collections import defaultdict 
from nltk.corpus import wordnet as wn
from gensim.matutils import sparse2full
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CORPUS_PATH = "data/baleen_sample"
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'


class PickledCorpus(CategorizedCorpusReader, CorpusReader):
    
    def __init__(self, root, fileids=PKL_PATTERN, cat_pattern=CAT_PATTERN):
        CategorizedCorpusReader.__init__(self, {"cat_pattern": cat_pattern})
        CorpusReader.__init__(self, root, fileids)
        
        self.punct = set(string.punctuation) | {'“', '—', '’', '”', '…'}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.wordnet = nltk.WordNetLemmatizer() 
    
    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories=categories)
        return fileids
    
    def lemmatize(self, token, tag):
        token = token.lower()
        
        if token not in self.stopwords:
            if not all(c in self.punct for c in token):
                tag =  {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0], wn.NOUN)
                return self.wordnet.lemmatize(token, tag)
    
    def tokenize(self, doc):
        # Expects a preprocessed document, removes stopwords and punctuation
        # makes all tokens lowercase and lemmatizes them. 
        return list(filter(None, [
            self.lemmatize(token, tag)
            for paragraph in doc 
            for sentence in paragraph 
            for token, tag in sentence 
        ]))
    
    def docs(self, fileids=None, categories=None):
        # Resolve the fileids and the categories
        fileids = self._resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield self.tokenize(pickle.load(f))


# The `PickledCorpus` is a Python class that reads a continuous stream of pickle files from disk. The files themselves are preprocessed documents from RSS feeds in various topics (and is actually just a small sample of the documents that are in the larger corpus). If you're interestd in the ingestion and curation of this corpus, see [baleen.districtdatalabs.com](http://baleen.districtdatalabs.com). 
# 
# Just to get a feel for this data set, I'll load the corpus and print out the number of documents per category:
# 

# Create the Corpus Reader
corpus = PickledCorpus(CORPUS_PATH)


# Count the total number of documents
total_docs = 0

# Count the number of documents per category. 
for category in corpus.categories():
    num_docs = sum(1 for doc in corpus.fileids(categories=[category]))
    total_docs += num_docs 
    
    print("{}: {:,} documents".format(category, num_docs))
    
print("\n{:,} documents in the corpus".format(total_docs))


# Our corpus reader object handles text preprocessing with NLTK (the natural language toolkit), namely by converting each document as follows:
# 
# - tokenizing the document 
# - making all tokens lower case 
# - removes stopwords and punctuation 
# - converts words to their lemma 
# 
# Here is an example document:
# 

fid = random.choice(corpus.fileids())
doc = next(corpus.docs(fileids=[fid]))
print(" ".join(doc))


# The next step is to convert these documents into vectors so that we can apply machine learning. We'll use a bag-of-words (bow) model with TF-IDF, implemented by the Gensim library.
# 

# Create the lexicon from the corpus 
lexicon = gensim.corpora.Dictionary(corpus.docs())

# Create the document vectors 
docvecs = [lexicon.doc2bow(doc) for doc in corpus.docs()]

# Train the TF-IDF model and convert vectors to TF-IDF
tfidf = gensim.models.TfidfModel(docvecs, id2word=lexicon, normalize=True)
tfidfvecs = [tfidf[doc] for doc in docvecs]

# Save the lexicon and TF-IDF model to disk.
lexicon.save('data/topics/lexicon.dat')
tfidf.save('data/topics/tfidf_model.pkl')


# Documents are now described by the words that are most important to that document relative to the rest of the corpus. The document above has been transformed into the following vector with associated weights: 
# 

# Covert random document from above into TF-IDF vector 
dv = tfidf[lexicon.doc2bow(doc)]

# Print the document terms and their weights. 
print(" ".join([
    "{} ({:0.2f})".format(lexicon[tid], score)
    for tid, score in sorted(dv, key=itemgetter(1), reverse=True)
]))


# ### Topic Visualization with LDA
# 
# We have a lot of documents in our corpus, so let's see if we can cluster them into related topics using the Latent Dirichlet Model that comes with Gensim. This model is widely used for "topic modeling" -- that is clustering on documents. 
# 

# Select the number of topics to train the model on.
NUM_TOPICS = 10 

# Create the LDA model from the docvecs corpus and save to disk.
model = gensim.models.LdaModel(docvecs, id2word=lexicon, alpha='auto', num_topics=NUM_TOPICS)
model.save('data/topics/lda_model.pkl')


# Each topic is represented as a vector - where each word is a dimension and the probability of that word beloning to the topic is the value. We can use the model to query the topics for a document, our random document from above is assigned the following topics with associated probabilities:
# 

model[lexicon.doc2bow(doc)]


# We can assign the most probable topic to each document in our corpus by selecting the topic with the maximal probability: 
# 

topics = [
    max(model[doc], key=itemgetter(1))[0]
    for doc in docvecs
]


# Topics themselves can be described by their highest probability words:
# 

for tid, topic in model.print_topics():
    print("Topic {}:\n{}\n".format(tid, topic))


# We can plot each topic by using decomposition methods (TruncatedSVD in this case) to reduce the probability vector for each topic into 2 dimensions, then size the radius of each topic according to how much probability documents it contains donates to it. Also try with PCA, explored below!
# 

# Create a sum dictionary that adds up the total probability 
# of each document in the corpus to each topic. 
tsize = defaultdict(float)
for doc in docvecs:
    for tid, prob in model[doc]:
        tsize[tid] += prob


# Create a numpy array of topic vectors where each vector 
# is the topic probability of all terms in the lexicon. 
tvecs = np.array([
    sparse2full(model.get_topic_terms(tid, len(lexicon)), len(lexicon)) 
    for tid in range(NUM_TOPICS)
])


# Import the model family 
from sklearn.decomposition import TruncatedSVD 

# Instantiate the model form, fit and transform 
topic_svd = TruncatedSVD(n_components=2)
svd_tvecs = topic_svd.fit_transform(tvecs)


# Create the Bokeh columnar data source with our various elements. 
# Note the resize/normalization of the topics so the radius of our
# topic circles fits int he graph a bit better. 
tsource = ColumnDataSource(
        data=dict(
            x=svd_tvecs[:, 0],
            y=svd_tvecs[:, 1],
            w=[model.print_topic(tid, 10) for tid in range(10)],
            c=brewer['Spectral'][10],
            r=[tsize[idx]/700000.0 for idx in range(10)],
        )
    )

# Create the hover tool so that we can visualize the topics. 
hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
        ]
    )


# Create the figure to draw the graph on. 
plt = figure(
    title="Topic Model Decomposition", 
    width=960, height=540, 
    tools="pan,box_zoom,reset,resize,save"
)

# Add the hover tool 
plt.add_tools(hover)

# Plot the SVD topic dimensions as a scatter plot 
plt.scatter(
    'x', 'y', source=tsource, size=9,
    radius='r', line_color='c', fill_color='c',
    marker='circle', fill_alpha=0.85,
)

# Show the plot to render the JavaScript 
show(plt)


# ### Corpus Visualization with PCA
# 
# The bag of words model means that every token (string representation of a word) is a dimension and a document is represented by a vector that maps the relative weight of that dimension to the document by the TF-IDF metric. In order to visualize documents in this high dimensional space, we must use decomposition methods to reduce the dimensionality to something we can plot. 
# 
# One good first attempt is toi use principle component analysis (PCA) to reduce the data set dimensions (the number of vocabulary words in the corpus) to 2 dimensions in order to map the corpus as a scatter plot. 
# 
# We'll use the Scikit-Learn PCA transformer to do this work:
# 

# In order to use Scikit-Learn we need to transform Gensim vectors into a numpy Matrix. 
docarr = np.array([sparse2full(vec, len(lexicon)) for vec in tfidfvecs])


# Import the model family 
from sklearn.decomposition import PCA 

# Instantiate the model form, fit and transform 
tfidf_pca = PCA(n_components=2)
pca_dvecs = topic_svd.fit_transform(docarr)


# We can now use Bokeh to create an interactive plot that will allow us to explore documents according to their position in decomposed TF-IDF space, coloring by their topic. 
# 

# Create a map using the ColorBrewer 'Paired' Palette to assign 
# Topic IDs to specific colors. 
cmap = {
    i: brewer['Paired'][10][i]
    for i in range(10)
}

# Create a tokens listing for our hover tool. 
tokens = [
    " ".join([
        lexicon[tid] for tid, _ in sorted(doc, key=itemgetter(1), reverse=True)
    ][:10])
    for doc in tfidfvecs
]

# Create a Bokeh tabular data source to describe the data we've created. 
source = ColumnDataSource(
        data=dict(
            x=pca_dvecs[:, 0],
            y=pca_dvecs[:, 1],
            w=tokens,
            t=topics,
            c=[cmap[t] for t in topics],
        )
    )

# Create an interactive hover tool so that we can see the document. 
hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
            ("Topic", "@t"),
        ]
    )

# Create the figure to draw the graph on. 
plt = figure(
    title="PCA Decomposition of BoW Space", 
    width=960, height=540, 
    tools="pan,box_zoom,reset,resize,save"
)

# Add the hover tool to the figure 
plt.add_tools(hover)

# Create the scatter plot with the PCA dimensions as the points. 
plt.scatter(
    'x', 'y', source=source, size=9,
    marker='circle_x', line_color='c', 
    fill_color='c', fill_alpha=0.5,
)

# Show the plot to render the JavaScript 
show(plt)


# Another approach is to use the TSNE model for stochastic neighbor embedding. This is a very popular text clustering visualization/projection mechanism.
# 

# Import the TSNE model family from the manifold package 
from sklearn.manifold import TSNE 
from sklearn.pipeline import Pipeline

# Instantiate the model form, it is usually recommended 
# To apply PCA (for dense data) or TruncatedSVD (for sparse)
# before TSNE to reduce noise and improve performance. 
tsne = Pipeline([
    ('svd', TruncatedSVD(n_components=75)),
    ('tsne', TSNE(n_components=2)),
])
                     
# Transform our TF-IDF vectors.
tsne_dvecs = tsne.fit_transform(docarr)


# Create a map using the ColorBrewer 'Paired' Palette to assign 
# Topic IDs to specific colors. 
cmap = {
    i: brewer['Paired'][10][i]
    for i in range(10)
}

# Create a tokens listing for our hover tool. 
tokens = [
    " ".join([
        lexicon[tid] for tid, _ in sorted(doc, key=itemgetter(1), reverse=True)
    ][:10])
    for doc in tfidfvecs
]

# Create a Bokeh tabular data source to describe the data we've created. 
source = ColumnDataSource(
        data=dict(
            x=tsne_dvecs[:, 0],
            y=tsne_dvecs[:, 1],
            w=tokens,
            t=topics,
            c=[cmap[t] for t in topics],
        )
    )

# Create an interactive hover tool so that we can see the document. 
hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
            ("Topic", "@t"),
        ]
    )

# Create the figure to draw the graph on. 
plt = figure(
    title="TSNE Decomposition of BoW Space", 
    width=960, height=540, 
    tools="pan,box_zoom,reset,resize,save"
)

# Add the hover tool to the figure 
plt.add_tools(hover)

# Create the scatter plot with the PCA dimensions as the points. 
plt.scatter(
    'x', 'y', source=source, size=9,
    marker='circle_x', line_color='c', 
    fill_color='c', fill_alpha=0.5,
)

# Show the plot to render the JavaScript 
show(plt)


# # Day 01 - Visual Data Exploration 
# 
# Today's workshop will cover the following topics:
# 
# - Intro to matplotlib for visualization (the pyplot API)
# - Data Frames for visual exploration
# - Pandas plotting API
# - Seaborn for visual statistical analysis
# 
# The schedule is as follows: 
# 
# - Introduction to matplotlib (35 mins) 
# - Introduction to Pandas (35 mins)
# - Introduction to Seaborn (30 mins) 
# - Guided Workshop (40 mins) 
# 

# ## matplotlib 
# 
# matplotlib is a 2D Python plotting library that is meant to create _publication quality figures_ in both hardcopy and interactive environments. It is the cornerstone of data visualization in Python and as a result is a fiscally sponsored project of the [NumFocus](http://www.numfocus.org/) organization. matplotlib is: 
# 
# - Open Source and Free 
# - Platform Agnostic
# - A General Visual Framework 
# - Optimized for Scientific Visualization 
# 
# The primary way to interact with matplotlib is through the `pyplot` API, which replaced the more proceedural `pylab` API which was intended to emulate MATLAB graphics commands. (**Note:** the `pylab` API is no longer supported and you shouldn't use it). The `pyplot` API is a _simple_ interface to the drawing components provided by matplotlib, as shown in the component architecture below:
# 
# ![The matplotlib Component Model](figures/matplotlib_components.png)
# 
# <p><center><small>The above figure is from McGreggor, Duncan M. _Mastering matplotlib_. Packt Publishing Ltd, 2015.</small></center></p>
# 
# In this notebook we'll explore using matplotlib in noteboook mode. This allows simple interaction of the kind that is provided in the Tk window mode, which I'll also demonstrate. In order to set matplotlib's mode, we must use the Jupyter notebook magic functions:
# 
#     %matplotlib notebook 
#     %matplotlib inline 
#     
# Note: the matplotlib inline function simply renders the figure created by the cell as a static image. This is useful for notebooks who you expect to "run all" cells routinely. Note also that this affects both Pandas and Seaborn rendering as well. 
# 
# The best thing for the magic function to work is to have it first thing in the notebook, we'll also import our packages into the namespace using the standard short names:

get_ipython().magic('matplotlib notebook')

import numpy as np
import matplotlib.pyplot as plt 


# matplotlib comes with default functions that allow customizing nearly every part of the figure. This is both a benefit and a barrier: you have fine grained control at the cost of copmlexity. Generally speaking the matplotlib defaults are good, considering visibility for colorblind folks, printing in black and white, etc. Importantly in the upcoming 2.0 release, the styles are going to be updated to look even better. 
# 
# The basic interaction of matplotlib is to pass data to functions in the `plt` module:
# 

# Create the X data points as a numpy array 
X = np.linspace(-10, 10, 255)

# Compute two quadratic functions 
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

plt.plot(X, Y1)
plt.plot(X, Y2)


# The `plt.plot` function does a lot of work on our behalf: it initializes the figure, creates a subplot with axes, then computes and draws two `Line2D` objects. Because we're in `notebook` mode, the current figure is displayed in an interactive fashion and can be explored until "shutdown". 
# 
# Key points:
# 
# - There is a global figure that is drawn on 
# - The objects that are returned from these functions can be directly manipulated 
# - Outside of a notebook nothing will be rendered until `plt.show` 
# 
# Just note, to get the same functionality in a Python script you'll have to tell `pyplot` to render the figure, either to an interactive backend like Tk or to an image, either raster or SVG:
# 
#     plt.show() 
#     plt.savefig('myfig.pdf') 
# 
# The goal of `pyplot` has always been to give as simple plotting functions as possible, so that figures can be drawn without much effort. In practice, it is easy to get simple graphs plotted, but much tougher to configure them as needed. 
# 
# Let's now look at all the steps it would take to draw this from scratch, which also demonstrate the configuration control you have over the figure.
# 

# Create a new figure of size 8x6 points, using 72 dots per inch 
plt.figure(figsize=(8,6), dpi=72)

# Create a new subplot from a 1x1 grid 
plt.subplot(111)

# Create the data to plot 
X = np.linspace(-10, 10, 255)
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 

# Plot the first quadratic using a blue color with a continuous line of 1px
plt.plot(X, Y1, color='blue', linewidth=1.0, linestyle='-')

# Plot the second quadratic using a green color with a continuous line of 1px
plt.plot(X, Y2, color='green', linewidth=1.0, linestyle='-')

# Set the X limits 
plt.xlim(-10, 10)

# Set the X ticks 
plt.xticks(np.linspace(-10, 10, 9, endpoint=True))

# Set the Y limits 
plt.ylim(0, 350)

# Set the Y ticks 
plt.yticks(np.linspace(0, 350, 5, endpoint=True))

# Save the figure to disk 
plt.savefig("figures/quadratics.png")


# Create the data to plot 
# This data will be referenced for the next plots below
# For Jupyter notebooks, pay attention to variables! 

X = np.linspace(-10, 10, 255)
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50 


# We'll look at each of these steps in detail in the next few boxes.
# 
# ### Colors and style 
# 
# We can directly pass colors and style to each of the drawing functions in the `pyplot` API. The arguments for color and linestyle can either be full words, e.g. "blue" or "dashed" or they can be shortcodes, for example 'b' or '--'. 
# 
# The color cycle in matplotlib determines which colors will be used for each new element drawn to the graph. The cycle is keyed to the short codes: 'bgrmyck' which stands for:
# 
#     blue green red maroon yellow cyan key 
#     
# A quick visualization of these colors is as follows:
# 

from matplotlib.colors import ListedColormap

colors = 'bgrmyck'
fig, ax = plt.subplots(1, 1, figsize=(7, 1))
ax.imshow(np.arange(7).reshape(1,7), cmap=ListedColormap(list(colors)), interpolation="nearest", aspect="auto")
ax.set_xticks(np.arange(7) - .5)
ax.set_yticks([-0.5,0.5])
ax.set_xticklabels([])
ax.set_yticklabels([])


# The default style is currently 'ggplot' -- though this is going to be updated soon. You can set the style of the graphs, or even provide your own CSS style sheet with the `plt.style.use` function: 
# 

# plt.style.use('fivethirtyeight')

# Note that I'm going to use temporary styling so I don't mess up the notebook! 
with plt.style.context(('fivethirtyeight')):
    plt.plot(X, Y1)
    plt.plot(X, Y2)


# To see the available styles:
for style in plt.style.available: print("- {}".format(style))


# Note also that styles can be composed together by passing a list. Styles farther to the right will override styles to the left. 
# 
# Line styles can be set using the following shortcodes (note that marker styles for scatter plots can also be set using filled and point shortcodes):
# 
# ![Line Styles](figures/linestyles.png)
# 
# So back to our original graphs we can convert the figure to have different colors and styles:

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-")


# We can also change the x and y limits to put some space into out graph:
# 

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(Y1.min()*-1.1, Y2.max()*1.1)


# We can add a legend and a title:
# 

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-", label="Y1")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-", label="Y2")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(Y1.min()*-1.1, Y2.max()*1.1)

plt.title("Two Quadratic Curves")
plt.legend(loc='best')


# And annotate some po ints on our graph:
# 

plt.figure(figsize=(9,6))
plt.plot(X, Y1, color="b", linewidth=2.5, linestyle="-", label="Y1")
plt.plot(X, Y2, color="r", linewidth=2.5, linestyle="-", label="Y2")

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(0, Y2.max()*1.1)

plt.title("Two Quadratic Curves")
plt.legend(loc='best')

# Annotate the blue line 
x = 6 
y = 2*x ** 2 + 10
plt.plot([x,x], [0, y], color='blue', linewidth=1.5, linestyle='--')
plt.scatter([x,], [y,], color='blue', s=50, marker='D')

plt.annotate(
    r'$2x^2+10={}$'.format(y), xy=(x,y), xycoords='data', xytext=(10,-50), 
    fontsize=16, textcoords='offset points',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)

# Annotate the red line
x = -3
y = 3*x ** 2 + 50
plt.plot([x,x], [0, y], color='red', linewidth=1.5, linestyle='--')
plt.scatter([x,], [y,], color='red', s=50, marker='s')

plt.annotate(
    r'$3x^2+50={}$'.format(y), xy=(x,y), xycoords='data', xytext=(10,50), 
    fontsize=16, textcoords='offset points',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)


# ### Figures, Subplots, and Axes 
# 
# The figure is the GUI window that displays the entirety of the drawing. Figures are numbered starting from 1, and each new plot has its own figure. Generally speaking though you're only working on one global figure at a time. Figures have several properties that can be configured:
# 
# - **num**: the number of the figure. 
# - **figsize**: the size of the figure in inches (width, height)
# - **dpi**: resolution in dots per inch
# - **facecolor**: the color of the drawing background 
# - **edgecolor**: the color of the edge around the drawing background
# - **frameon**: draw the figure frame or not 
# 
# Subplots allow you to arrange plots in a rectangular grid. They are specified by the rows and columns as well as the number of the plot (e.g. it's id). The [gridspec](http://matplotlib.sourceforge.net/users/gridspec.html) command gives a much more controlled alternative. 
# 
# ![horizontal](figures/subplot-horizontal.png)
# 
# ![vertical](figures/subplot-vertical.png)
# 
# ![grid](figures/subplot-grid.png)
# 
# Axes are very similar to subplots but allow placement of plots at any location in the figure. This allows more fine grained plot within a plot control, but also the addition of complex images, for example colorbars for heatmaps. Because axes are the primary drawing space, they can also be worked on directly, usually with the `set_[]` style command. 
# 
# ![axes](figures/axes.png)
# 
# ![axes](figures/axes-2.png)
# 
# There are many more commands, more than we can cover in the section. But we'll view them more specifically using Pandas and Seaborn. 

# ## Pandas
# 
# Pandas is an open source Python library that provides high performance _data structures_ for data analysis, in particular the `Series` and `DataFrame` objects. The focus today is not on Pandas, however, but rather on its plotting library. 
# 
# Pandas' plotting library is essentially a wrapper around matplotlib that uses information from the `DataFrame` and series objects in order to provide more detail. 
# 

import pandas as pd


# Create a random timeseries object 
ts = pd.Series(np.random.randn(365), index=pd.date_range('1/1/2010', periods=365))
ts = ts.cumsum()
ts.plot() 


# Note that `series.plot` and `df.plot` are not exactly the same thing as matplotlib, but in many cases do take the exact same arguments. Let's look at a few Pandas's specific arguments:
# 

df = pd.DataFrame(np.random.randn(365, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
df.plot();


df = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()
df['A'] = pd.Series(list(range(len(df))))
df.plot(x='A', y='B')


# The `kind` argument allows you to specify a wide variety of plots for data:
# 
# - ‘bar’ or ‘barh’ for bar plots
# - ‘hist’ for histogram
# - ‘box’ for boxplot
# - ‘kde’ or 'density' for density plots
# - ‘area’ for area plots
# - ‘scatter’ for scatter plots
# - ‘hexbin’ for hexagonal bin plots
# - ‘pie’ for pie plots
# 

df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot(kind='bar')


df2.plot.bar(stacked=True)


df2.plot(kind='area')


df2.plot.area(stacked=False)


df3 = pd.DataFrame(np.random.rand(25, 4), columns=['a', 'b', 'c', 'd'])
ax = df3.plot.scatter(x='a', y='b', color='r', label="B")
df3.plot.scatter(x='a', y='c', color='c', ax=ax, label="C")
df3.plot.scatter(x='a', y='d', color='g', ax=ax, label="D")


# Add new dimensions such as color and size based on other attributes in the data frame
# This creates a bubble plot, points sized based on the 'C' attribute and colors based on 'D'.
df3.plot.scatter(x='a', y='b', c=df3['d'], s=df3['c']*200);


# Hexbin plots can be a useful alternative to scatter plots if your data are too dense to plot each point individually.
# 

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)
df.plot.hexbin(x='a', y='b', gridsize=25)


# ### Higher Dimensional Data Visualization 
# 
# We've already seen that attributes of the plot can allow us to increase the amount of information we add; however we typically say that only 7 attributes can be embedded into a plot:
# 
# - space (possibly in three dimensions)
# - color
# - size
# - shape 
# - time (slider)
# 
# Pandas does provide some interesting visualizations for higher dimensional work, though, including SPLOMs, Radviz, and Parallel coordinates
# 

# Load the data
import os 

DATA = os.path.join("data", "wheat", "seeds_dataset.txt")

FEATURES  = [
    "area",
    "perimeter",
    "compactness",
    "length",
    "width",
    "asymmetry",
    "groove",
    "label"
]

LABEL_MAP = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian",
}

# Read the data into a DataFrame
df = pd.read_csv(DATA, sep='\s+', header=None, names=FEATURES)

# Convert class labels into text
for k,v in LABEL_MAP.items():
    df.ix[df.label == k, 'label'] = v

# Describe the dataset
print(df.describe())


# Determine the shape of the data
print("{} instances with {} features\n".format(*df.shape))

# Determine the frequency of each class
print(df.groupby('label')['label'].count())


from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(9,9), diagonal='kde')


from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(9,9))
parallel_coordinates(df, 'label')


from pandas.tools.plotting import radviz
plt.figure(figsize=(9,9))
radviz(df, 'label')


# ## Seaborn 
# 
# Seaborn is a _statistical_ visualization library, meaning that it embeds both models and data into visualizations, and is intended for visual statistical analysis. It is currently free and open source, written by Michael Waskom while he was at Stanford university. 
# 
# Similar to Pandas, Seaborn wraps matplotlib, and even wraps data frames - but provides higher level abstractions for a bunch of different plots and configurations. 
# 

get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import seaborn as sns

IRIS = os.path.join("data", "iris.csv")
data = pd.read_csv(IRIS)

sns.set_style('darkgrid')
sns.set_palette('deep')
sns.set_context('notebook')


sns.pairplot(data, hue='class', diag_kind="kde", size=3)


sns.distplot(data['sepal width'], rug=True)


sns.jointplot("petal length", "petal width", data=data, kind='reg', size=6)


sns.boxplot(x='petal length', data=data)


sns.boxplot(data=data)


sns.violinplot(data=data)


sns.swarmplot(data=data)


ax = sns.boxplot(data=data)
ax = sns.swarmplot(data=data)


sns.lmplot(x="sepal width", y="sepal length", hue="class", data=data)


sns.lmplot(x="sepal width", y="sepal length", col="class", data=data)


sns.barplot(data=data)


import numpy as np
sns.barplot(data=data, estimator=np.median)


# # Exception Handling 
# 
# This notebook is intended to demonstrate the basics of exception handling and the use of context management in order to handle standard cases. I'm hoping that notes can be live and editable to create a set of documentation for you to use as you're learning Python. 
# 
# ## Exceptions 
# 
# **Exceptions** are a tool that programmers use to describe errors or faults that are _fatal_ to the program; e.g. the program cannot or should not continue when an exception occurs. Exceptions can occur due to programming errors, user errors, or simply unexpected conditions like no internet access. Exceptions themselves are simply objects that contain information about what went wrong. Exceptions are usually defined by their `type` - which describes broadly the class of exception that occurred, and by a `message` that says specifically what happened. Here are a few common exception types:
# 
# - `SyntaxError`: raised when the programmer has made a mistake typing Python code correctly. 
# - `AttributeError`: attempting to access an attribute on an object that does not exist 
# - `KeyError`: attempting to access a key in a dictionary that does not exist 
# - `TypeError`: raised when an argument to a function is not the right type (e.g. a `str` instead of `int`) 
# - `ValueError`: when an argument to a function is the right type but not in the right domain (e.g. an empty string)
# - `ImportError`: raised when an import fails 
# - `IOError`: raised when Python cannot access a file correctly on disk 
# 
# Exceptions are defined in a class hierarchy - e.g. every exception is an object whose class defines it's type. The base class is the `Exception` object. All `Exception` objects are initialized with a message - a string that describes exactly what went wrong. Constructed objects can then be "raised" or "thrown" with the `raise` keyword:
# 
# ```python
# raise Exception("Something bad happened!") 
# ```
# 
# The reason the keyword is `raise` is because Python program execution creates what's called a "stack" as functions call other functions, which call other functions, etc. When a function (at the bottom of the stack) raises an Exception, it is propagated up through the call stack so that every function gets a chance to "handle" the exception (more on that later). If the exception reaches the top of the stack, then the program terminates and a _traceback_ is printed to the console. The traceback is meant to help developers identify what went wrong in their code. 
# 
# Let's take a look at a simple example:
# 

def first(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Call sub steps in order 
    step = first_task_one(step, badstep)
    step = first_task_two(step, badstep)
    
    # Return the step that we're on 
    return step 


def first_task_one(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Call sub steps in order 
    step = first_task_one_subtask_one(step, badstep)
    
    # Return the step that we're on 
    return step 


def first_task_one_subtask_one(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Return the step that we're on 
    return step 


def first_task_two(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Return the step that we're on 
    return step 


def second(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Call sub steps in order 
    step = second_task_one(step, badstep)
    
    # Return the step that we're on 
    return step 


def second_task_one(step, badstep=None):
    # Increment the step 
    step += 1 
    
    # Check if this is a bad step 
    if badstep == step:
        raise ValueError("Failed after {} steps".format(step))
    
    # Return the step that we're on 
    return step 

def main(badstep=None, **kwargs):
    """
    This function is the entry point of the program, it does 
    work on the arguments by calling each step function, which 
    in turn call substep functions. 
    
    Passing in a number for badstep will cause whichever step 
    that is to raise an exception. 
    """
    
    step = 0 # count the steps 
    
    # Execute each step one at a time. 
    step = first(step, badstep) 
    step = second(step, badstep)
    
    # Return a report 
    return "Sucessfully executed {} steps".format(step) 

if __name__ == "__main__":
    main()


# The above example represents a fairly complex piece of code that has lots of functions that call lots of other functions. The question is then, _how do we know where our code went wrong?_ The answer is the _traceback_ - which deliniates exactly the functions that the exception was raised through. Let's trigger the exception and the traceback:
# 

main(3)


# The way to read the traceback is to start at the very bottom. As you can see it indicates the type of the exception, followed by a colon, and then the message that was passed to the exception constructor. Often, this information is enough to figure out what is going wrong. However, if we're unsure where the problem occurred, we can step back through the traceback in a bottom to top fashion. 
# 
# The first part of the traceback indicates the exact line of code and file where the exception was raised, as well as the name of the function it was raised in. If you called `main(3)` than this indicates that `first_task_one_subtask_one` is the function where the problem occurred. If you wrote this function, then perhaps that is the place to change your code to handle the exception. 
# 
# However, many times you're using third party libraries or Python standard library modules, meaning the location of the exception raised is not helpful, since you can't change that code. Therefore, you will continue up the call stack until you discover a file/function in the code you wrote. This will provide the surrounding context for why the error was raised, and you can use `pdb` or even just `print` statements to debug the variables around that line of code. Alternatively you can simply handle the exception, which we'll discuss shortly. In the example above, we can see that `first_task_one_subtask_one` was called by `first_task_one` at line 46, which was called by `first` at line 30, which was called by `main` at line 14. 
# 
# ## Catching Exceptions 
# 
# If the exception was caused by a programming error, the developer can simply change the code to make it correct. However, if the exception was created by bad user input or by a bad environmental condition (e.g. the wireless is down), then you don't want to crash the program. Instead you want to provide feedback and allow the user to fix the problem or try again. Therefore in your code, you can catch exceptions at the place they occur using the following syntax:
# 
# ```python
# try:
#     # Code that may raise an exception 
# except AttributeError as e:
#     # Code to handle the exception case
# finally:
#     # Code that must run even if there was an exception 
# ```
# 
# What we're basically saying is `try` to do the code in the first block - hopefully it works. If it raises an `AttributeError` save that exception in a variable called `e` (the `as e` syntax) then we will deal with that exception in the `except` block. Then `finally` run the code in the `finally` block even if an exception occurs. By specifying exactly the type of exception we want to catch (`AttributeError` in this case), we will not catch all exceptions, only those that are of the type specified, including subclasses. If we want to catch _all_ exceptions, you can use one of the following syntaxes:
# 
# ```python
# try:
#     # Code that may raise an exception 
# except:
#     # Except all exceptions 
# ```
# 
# or 
# 
# ```python
# try:
#     # Code that may raise an exception 
# except Exception as e:
#     # Except all exceptions and capture in variable e 
# ```
# 
# However, it is best practice to capture _only_ the type of exception you expect to happen, because you could accidentaly create the situation where you're capturing fatal errors but not handling them appropriately. Here is an example:
# 

import random 

class RandomError(Exception):
    """
    A custom exception for this code block. 
    """
    pass 


def randomly_errors(p_error=0.5):
    if random.random() <= p_error:
        raise RandomError("Error raised with {:0.2f} likelihood!".format(p_error))


try:
    randomly_errors(0.5) 
    print("No error occurred!")
except RandomError as e:
    print(e)
finally:
    print("This runs no matter what!")


# This code snippet demonstrates a couple of things. First you can define your own, program-specific exceptions by defining a class that extends `Exception`. We have done so and created our own `RandomError` exception class. Next we have a function that raises a `RandomError` with some likelihood which is an argument to the function. Then we have our exception handling block that calls the function and handles it. 
# 
# Try the following the code snippet:
# 
# - Change the likelihood of the error to see what happens 
# - except `Exception` instead of `RandomError`
# - except `TypeError` instead of `RandomError` 
# - Call `randomly_errors` again inside of the `except` block 
# - Call `randomly_errors` again inside of the `finally` block
# 
# Make sure you run the code multiple times since the error does occur randomly! 
# 
# ## LBYL vs. EAFP 
# 
# One quick note on exception handling in Python. You may wonder why you must use a `try/except` block to handle exceptions, couldn't you simply do a check that the exception won't occur before it does? For example, consider the following code:
# 
# ```python
# if key in mydict:
#     val = mydict[key] 
#     # Do something with val 
# else:
#     # Handle the fact that mydict doesn't have a required key. 
# ```
# 
# This code checks if a key exists in the dictionary before using it, then uses an else block to handle the "exception". This is an alternative to the following code:
# 
# ```python
# try:
#     val = mydict[key]
#     # Do something with val 
# except KeyError:
#     # Handle the fact that mydict doesn't have a required key. 
# ```
# 
# Both blocks of code are valid. In fact they have names:
# 
# 1. Look Before You Leap (LBYL)
# 2. Easier to Ask Forgiveness than Permission (EAFP) 
# 
# For a variety of reasons, the second example (EAFP) is more _pythonic_ &mdash; that is the prefered Python Syntax, commonly accepted by Python developers. For more on this, please see Alex Martelli's excellent PyCon 2016 talk, [Exception and error handling in Python 2 and Python 3](https://www.youtube.com/watch?v=frZrBgWHJdY). 
# 
# ## Context Management 
# 
# Python does provide a syntax for embedding common `try/except/finally` blocks in an easy to read format called context management. To motivate the example, consider the following code snippet:
# 
# ```python
# try:
#     fobj = open('path/to/file.txt, 'r') 
#     data = fobj.read() 
# except FileNotFoundError as e:
#     print(e)
#     print("Could not find the necessary file!) 
# finally:
#     fobj.close() 
# ```
# 
# This is a very common piece of code that opens a file and reads data from it. If the file doesn't exist, we simply alert the user that the required file is missing. No matter what, the file is closed. This is critical because if the file is not closed properly, it can be corrupted or not available to other parts of the program. Data loss is not acceptable, so we need to ensure that no matter what the file is closed when we're done with it. So we can do the following:
# 
# ```python
# with open('path/to/file.txt', 'r') as fobj:
#     data = fobj.read() 
# ```
# 
# The `with as` syntax implements context management. On `with`, a function called the `enter` function is called to do some work on behalf of the user (in this case open a file), and the return of that function is saved in the `fobj` variable. When this block is complete, the finally is called by implementing an `exit` function. (Note that the `except` part is not implemented in this particular code). In this way, we can ensure that the `try/finally` for opening and reading files is correctly implemented. 
# 
# Writing your own context managers is possible, but beyond the scope of this note. Suffice it to say, you should always use the `with/as` syntax for opening files!
# 

get_ipython().magic('matplotlib inline')


# # BLS Timeseries Data Exploration 
# 
# In this workbook, I've set up a data frame of Bureau of Labor Statistics time series data, your goal is to explore and visualize the time series data using pandas, matplotlib, seaborn, or even Bokeh!
# 

# Imports 
import csv 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from itertools import groupby
from operator import itemgetter


# ## Data Loading 
# 
# The data is stored in a zip file in the data directory called `data/bls.zip` -- unzip this file and there are two CSV files. The first `series.csv` is a description of the various time series that are in the data frame. The second complete record of the time series data, with the associated time series id. 
# 
# We we load the series information into it's own dataframe for quick lookup (like a database) and then create a dataframe of each individual series data, identified by their ID. There is more information in the CSV, which you can explore if you'd like. 
# 

# Load the series data 
info = pd.read_csv('../data/bls/series.csv')

def series_info(blsid, info=info):
    return info[info.blsid == blsid]

# Use this function to lookup specific BLS series info. 
series_info("LNS14000025")


# Load each series, grouping by BLS ID
def load_series_records(path='../data/bls/records.csv'):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        
        for blsid, rows in groupby(reader, itemgetter('blsid')):
            # Read all the data from the file and sort 
            rows = list(rows) 
            rows.sort(key=itemgetter('period'))
            
            # Extract specific data from each row, namely:
            # The period at the month granularity 
            # The value as a float 
            periods = [pd.Period(row['period']).asfreq('M') for row in rows]
            values = [float(row['value']) for row in rows]
            
            yield pd.Series(values, index=periods, name=blsid)
            

series = pd.concat(list(load_series_records()), axis=1)
series


get_ipython().magic('matplotlib inline')


# # Pairwise Ranking of Features
# 
# ![Rank 1D Histogram](../figures/rank_1d_hist.png)
# 
# ![Rank 1D Histogram](../figures/rank_1d_box.png)
# 
# ![Rank 1D Histogram](../figures/rank_2d.png)
# 
# ![Rank 1D Histogram](../figures/joint.png)

# Imports 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from ipywidgets import interact, interactive, fixed

import ipywidgets as widgets


# Data Loading 
columns = OrderedDict([
    ("DAY", "the day of data collection"),
    ("Q-E", "input flow to plant"),
    ("ZN-E", "input Zinc to plant"),
    ("PH-E", "input pH to plant"),
    ("DBO-E", "input Biological demand of oxygen to plant"),
    ("DQO-E", "input chemical demand of oxygen to plant"),
    ("SS-E", "input suspended solids to plant"),
    ("SSV-E", "input volatile supended solids to plant"),
    ("SED-E", "input sediments to plant"),
    ("COND-E", "input conductivity to plant"),
    ("PH-P", "input pH to primary settler"),
    ("DBO-P", "input Biological demand of oxygen to primary settler"),
    ("SS-P", "input suspended solids to primary settler"),
    ("SSV-P", "input volatile supended solids to primary settler"),
    ("SED-P", "input sediments to primary settler"),
    ("COND-P", "input conductivity to primary settler"),
    ("PH-D", "input pH to secondary settler"),
    ("DBO-D", "input Biological demand of oxygen to secondary settler"),
    ("DQO-D", "input chemical demand of oxygen to secondary settler"),
    ("SS-D", "input suspended solids to secondary settler"),
    ("SSV-D", "input volatile supended solids to secondary settler"),
    ("SED-D", "input sediments to secondary settler"),
    ("COND-S", "input conductivity to secondary settler"),
    ("PH-S", "output pH"),
    ("DBO-S", "output Biological demand of oxygen"),
    ("DQO-S", "output chemical demand of oxygen"),
    ("SS-S", "output suspended solids"),
    ("SSV-S", "output volatile supended solids"),
    ("SED-S", "output sediments"),
    ("COND-", "output conductivity"),
    ("RD-DB-P", "performance input Biological demand of oxygen in primary settler"),
    ("RD-SSP", "performance input suspended solids to primary settler"),
    ("RD-SE-P", "performance input sediments to primary settler"),
    ("RD-DB-S", "performance input Biological demand of oxygen to secondary settler"),
    ("RD-DQ-S", "performance input chemical demand of oxygen to secondary settler"),
    ("RD-DB-G", "global performance input Biological demand of oxygen"),
    ("RD-DQ-G", "global performance input chemical demand of oxygen"),
    ("RD-SSG", "global performance input suspended solids"),
    ("RD-SED-G", "global performance input sediments"),
])

data = pd.read_csv("../data/water-treatment.data", names=columns.keys())
data = data.replace('?', np.nan)


# Capture only the numeric columns in the data set. 
numeric_columns = [col for col in columns.keys() if col != "DAY"]
data = data[numeric_columns].apply(pd.to_numeric)


# ## 2D Rank Features 
# 

def apply_column_pairs(func):
    """
    Applies a function to a pair of columns and returns a new 
    dataframe that contains the result of the function as a matrix
    of each pair of columns. 
    """
    
    def inner(df):
        cols = pd.DataFrame([
            [
                func(df[acol], df[bcol]) for bcol in df.columns
            ] for acol in df.columns
        ])

        cols.columns = df.columns
        cols.index = df.columns 
        return cols

    return inner 


@apply_column_pairs
def least_square_error(cola, colb):
    """
    Computes the Root Mean Squared Error of a linear regression 
    between two columns of data. 
    """
    x = cola.fillna(np.nanmean(cola))
    y = colb.fillna(np.nanmean(colb))
    
    m, b = np.polyfit(x, y, 1)
    yh  = (x * m) + b 
    return ((y-yh) ** 2).mean()


labeled_metrics = {
    'Pearson': 'pearson', 
    'Kendall Tao': 'kendall', 
    'Spearman': 'spearman', 
    'Pairwise Covariance': 'covariance',
    'Least Squares Error': 'lse', 
}

@interact(metric=labeled_metrics, data=fixed(data))
def rank2d(data, metric='pearson'):
    """
    Creates a visualization of pairwise ranking by column in the data. 
    """
    
    # The different rank by 2d metrics. 
    metrics = {
        "pearson": lambda df: df.corr('pearson'), 
        "kendall": lambda df: df.corr('kendall'), 
        "spearman": lambda df: df.corr('spearman'), 
        "covariance": lambda df: df.cov(), 
        "lse": least_square_error,
    }
    
    # Quick check to make sure a valid metric is passed in. 
    if metric not in metrics:
        raise ValueError(
            "'{}' not a valid metric, specify one of {}".format(
                metric, ", ".join(metrics.keys())
            )
        )
    
    
    # Compute the correlation matrix
    corr = metrics[metric](data)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title("{} metric across {} features".format(metric.title(), len(data.columns)))
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# # Classification Models with Scikit-Learn
# 
# **Classification** algorithms assign a discrete label to a vector, X based on the vector's elements. Unlike regression models, there are a wide array of classification models from probabilistic Bayesian and Logistic Regression models to non-parameteric methods like kNN and Random Forests to linear methods like SVMs and LDA. 
# 
# As before the basic methodology is to select the best model through trial and evaluation with cross-validation and F1 Scores. 
# 

# Using the IRIS data set - the classic classification data set. 
from sklearn.cross_validation import train_test_split as tts
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

data = load_iris()
X_train, X_test, y_train, y_test = tts(data.data, data.target)


# ## Non-Parametric Methods 
# 

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(classification_report(yhat, y_test))


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(classification_report(yhat, y_test))


# ## Probabalistic Methods 
# 

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(classification_report(yhat, y_test))


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(classification_report(yhat, y_test))


# ## Linear Methods 
# 

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(classification_report(yhat, y_test))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(classification_report(yhat, y_test))


# ## Classifying Wheat Kernels by Physical Property
# 
# Downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/seeds) on February 26, 2015. The first thing is to fully describe your data in a README file. The dataset description is as follows:
# 
# - Data Set: Multivariate
# - Attribute: Real
# - Tasks: Classification, Clustering
# - Instances: 210
# - Attributes: 7
# 
# ### Data Set Information:
# 
# The examined group comprised kernels belonging to three different varieties of wheat: Kama, Rosa and Canadian, 70 elements each, randomly selected for the experiment. High quality visualization of the internal kernel structure was detected using a soft X-ray technique. It is non-destructive and considerably cheaper than other more sophisticated imaging techniques like scanning microscopy or laser technology. The images were recorded on 13x18 cm X-ray KODAK plates. Studies were conducted using combine harvested wheat grain originating from experimental fields, explored at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin.
# 
# The data set can be used for the tasks of classification and cluster analysis.
# 
# ### Attribute Information:
# 
# To construct the data, seven geometric parameters of wheat kernels were measured:
# 
# 1. area A,
# 2. perimeter P,
# 3. compactness C = 4*pi*A/P^2,
# 4. length of kernel,
# 5. width of kernel,
# 6. asymmetry coefficient
# 7. length of kernel groove.
# 
# All of these parameters were real-valued continuous.
# 
# ### Relevant Papers:
# 
# M. Charytanowicz, J. Niewczas, P. Kulczycki, P.A. Kowalski, S. Lukasik, S. Zak, 'A Complete Gradient Clustering Algorithm for Features Analysis of X-ray Images', in: Information Technologies in Biomedicine, Ewa Pietka, Jacek Kawa (eds.), Springer-Verlag, Berlin-Heidelberg, 2010, pp. 15-24.
# 
# ## Data Exploration 
# 
# In this section we will begin to explore the dataset to determine relevant information.
# 

get_ipython().magic('matplotlib inline')

import os
import json
import time
import pickle
import requests


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"

def fetch_data(fname='data/wheat/seeds_dataset.txt'):
    """
    Helper method to retreive the ML Repository dataset.
    """
    response = requests.get(URL)
    outpath  = os.path.abspath(fname)
    with open(outpath, 'wb') as f:
        f.write(response.content)
    
    return outpath

# Fetch the data if required
# DATA = fetch_data()


FEATURES  = [
    "area",
    "perimeter",
    "compactness",
    "length",
    "width",
    "asymmetry",
    "groove",
    "label"
]

LABEL_MAP = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian",
}

# Read the data into a DataFrame
df = pd.read_csv(DATA, sep='\s+', header=None, names=FEATURES)

# Convert class labels into text
for k,v in LABEL_MAP.items():
    df.ix[df.label == k, 'label'] = v

# Describe the dataset
print(df.describe())


# Determine the shape of the data
print("{} instances with {} features\n".format(*df.shape))

# Determine the frequency of each class
print(df.groupby('label')['label'].count())


# Create a scatter matrix of the dataframe features
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.show()


from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(12,12))
parallel_coordinates(df, 'label')
plt.show()


from pandas.tools.plotting import radviz
plt.figure(figsize=(12,12))
radviz(df, 'label')
plt.show()


# ## Data Extraction 
# 
# One way that we can structure our data for easy management is to save files on disk. The Scikit-Learn datasets are already structured this way, and when loaded into a `Bunch` (a class imported from the `datasets` module of Scikit-Learn) we can expose a data API that is very familiar to how we've trained on our toy datasets in the past. A `Bunch` object exposes some important properties:
# 
# - **data**: array of shape `n_samples` * `n_features`
# - **target**: array of length `n_samples`
# - **feature_names**: names of the features
# - **target_names**: names of the targets
# - **filenames**: names of the files that were loaded
# - **DESCR**: contents of the readme
# 
# **Note**: This does not preclude database storage of the data, in fact - a database can be easily extended to load the same `Bunch` API. Simply store the README and features in a dataset description table and load it from there. The filenames property will be redundant, but you could store a SQL statement that shows the data load. 
# 
# In order to manage our data set _on disk_, we'll structure our data as follows:
# 

from sklearn.datasets.base import Bunch

DATA_DIR = os.path.abspath(os.path.join(".", "..", "data", "wheat"))

# Show the contents of the data directory
for name in os.listdir(DATA_DIR):
    if name.startswith("."): continue
    print("- {}".format(name))


def load_data(root=DATA_DIR):
    # Construct the `Bunch` for the wheat dataset
    filenames     = {
        'meta': os.path.join(root, 'meta.json'),
        'rdme': os.path.join(root, 'README.md'),
        'data': os.path.join(root, 'seeds_dataset.txt'),
    }

    # Load the meta data from the meta json
    with open(filenames['meta'], 'r') as f:
        meta = json.load(f)
        target_names  = meta['target_names']
        feature_names = meta['feature_names']

    # Load the description from the README. 
    with open(filenames['rdme'], 'r') as f:
        DESCR = f.read()

    # Load the dataset from the text file.
    dataset = np.loadtxt(filenames['data'])

    # Extract the target from the data
    data   = dataset[:, 0:-1]
    target = dataset[:, -1]

    # Create the bunch object
    return Bunch(
        data=data,
        target=target,
        filenames=filenames,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=DESCR
    )

# Save the dataset as a variable we can use.
dataset = load_data()

print(dataset.data.shape)
print(dataset.target.shape)


# ## Classification 
# 
# Now that we have a dataset `Bunch` loaded and ready, we can begin the classification process. Let's attempt to build a classifier with kNN, SVM, and Random Forest classifiers. 
# 

from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import KFold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def fit_and_evaluate(dataset, model, label, **kwargs):
    """
    Because of the Scikit-Learn API, we can create a function to
    do all of the fit and evaluate work on our behalf!
    """
    start  = time.time() # Start the clock! 
    scores = {'precision':[], 'recall':[], 'accuracy':[], 'f1':[]}
    
    for train, test in KFold(dataset.data.shape[0], n_folds=12, shuffle=True):
        X_train, X_test = dataset.data[train], dataset.data[test]
        y_train, y_test = dataset.target[train], dataset.target[test]
        
        estimator = model(**kwargs)
        estimator.fit(X_train, y_train)
        
        expected  = y_test
        predicted = estimator.predict(X_test)
        
        # Append our scores to the tracker
        scores['precision'].append(metrics.precision_score(expected, predicted, average="weighted"))
        scores['recall'].append(metrics.recall_score(expected, predicted, average="weighted"))
        scores['accuracy'].append(metrics.accuracy_score(expected, predicted))
        scores['f1'].append(metrics.f1_score(expected, predicted, average="weighted"))

    # Report
    print("Build and Validation of {} took {:0.3f} seconds".format(label, time.time()-start))
    print("Validation scores are as follows:\n")
    print(pd.DataFrame(scores).mean())
    
    # Write official estimator to disk
    estimator = model(**kwargs)
    estimator.fit(dataset.data, dataset.target)
    
    outpath = label.lower().replace(" ", "-") + ".pickle"
    with open(outpath, 'wb') as f:
        pickle.dump(estimator, f)

    print("\nFitted model written to:\n{}".format(os.path.abspath(outpath)))


# Perform SVC Classification
fit_and_evaluate(dataset, SVC, "Wheat SVM Classifier")


# Perform kNN Classification
fit_and_evaluate(dataset, KNeighborsClassifier, "Wheat kNN Classifier", n_neighbors=12)


# Perform Random Forest Classification
fit_and_evaluate(dataset, RandomForestClassifier, "Wheat Random Forest Classifier")


# # Natural Language Processing
# 
# In this notebook, we'll walk through some simple natural language processing techniques and work towards building a text classification model. Through this process we'll utilize the data science pipeline:
# 
# Ingestion &rarr; Wrangling &rarr; Analysis &rarr; Modeling &rarr; Visualization
# 
# The basic principle will be to fetch HTML data from web pages, then extract the text from it. We will then apply tokenization and tagging to the text to create a basic data structure. In preparation for modeling we'll normalize our text using lemmatization, then remove stopwords and punctuation. After that we'll vectorize our text, then send it to our classification model, which we will evaluate with cross validation. 
# 
# ## Preprocessing Text
# 
# 
# ### Step One: Fetch Data
# 
# For now, we'll simply ingest news articles from the Washington Post by looking up their ID from the short URL. 
# 

get_ipython().magic('matplotlib inline')


import os 
import requests 

WAPO = "http://wpo.st/"

def fetch_wapo(sid="ciSa2"):
    url = WAPO + sid 
    res = requests.get(url) 
    return res.text

story = fetch_wapo()


print(story)


# ### Step Two: Clean Up Data 
# 
# The HTML that we fetched contains navigation, advertisements, and markup not related to the text. We need to clean it up to extract only the part of the document we're interested in analyzing. 
# 
# Note that this is also the point that we should consider larger document structures like chapters, sections, or paragraphs. If we want to consider paragraphs, the `extract` function should return a list of strings that each represent a paragraph.
# 

from bs4 import BeautifulSoup
from readability.readability import Document

def extract(html):
    article = Document(html).summary()
    soup = BeautifulSoup(article, 'lxml')
    
    return soup.get_text()

story = extract(story)


print(story)


# ### Step Three: Tokenization
# 
# Tokenizers break down the text into units of logical meaning - sentences and words.
# 

import nltk 

def tokenize(text):
    for sent in nltk.sent_tokenize(text):
        yield list(nltk.word_tokenize(sent))

story = list(tokenize(story))


for sent in story: print(sent)


# ### Step Four: Tag Text 
# 
# Tagging adds information to the data structure we have -- namely the word class for each word (e.g. is it a Noun, Verb, Adjective, etc.). Note that tagging needs a complete sentence to work effectively. 
# 
# After we have tagged our text, we have completed the non-destructive operations on our text string, it is at this point that the text should be saved as a pickle to disk for use in downstream processing. 
# 

def tag(sents):
    for sent in sents:
        yield list(nltk.pos_tag(sent))

story = list(tag(story))


for sent in story: print(sent)


# ### Step 5: Normalize 
# 
# Normalization reduces the number of tokens that we pass to our analysis, allowing us to do more effective language inference. 
# 

from nltk.corpus import wordnet as wn

lemmatizer = nltk.WordNetLemmatizer()

def tagwn(tag):
    return {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)


def lemmatize(tagged_sents):
    for sent in tagged_sents:
        for token, tag in sent:
            yield lemmatizer.lemmatize(token, tagwn(tag))


story = list(lemmatize(story))


print(story)


from string import punctuation
from nltk.corpus import stopwords 

punctuation = set(punctuation)
stopwords = set(stopwords.words('english'))

def normalize(tokens):
    for token in tokens:
        token = token.lower()
        if not all(char in punctuation for char in token):
            if token not in stopwords:
                yield token
        

story = list(normalize(story))


print(story)


# ## Creating a Corpus
# 
# Building models requires gathering multiple documents and performing the processing steps on them that we showed above. We've used a tool called [Baleen](http://baleen.districtdatalabs.com/) to ingest data from RSS feeds for the past year. (It currently contains 1,154,100 posts for 373 feeds after 5,566 jobs). 
# 
# We've provided a small sample of the corpus to start playing with the tool. It has saved documents in the following structure:
# 
# - Each file stored in the directory of its category 
# - One document per file, stored as a pickle 
# - Document is a list of paragraphs 
# - Paragraph is a list of sentences 
# - Sentence is a list of (token, tag) tuples
# 
# We can then create a reader to automatically fetch data from our corpus. This is a bit more complex, but necessary. Also note that we add our normalization process here as well, just so we don't have to repeat steps later on. 
# 

import string
import pickle 

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CORPUS_PATH = "data/baleen_sample"
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

class PickledCorpus(CategorizedCorpusReader, CorpusReader):
    
    def __init__(self, root, fileids=PKL_PATTERN, cat_pattern=CAT_PATTERN):
        CategorizedCorpusReader.__init__(self, {"cat_pattern": cat_pattern})
        CorpusReader.__init__(self, root, fileids)
        
        self.punct = set(string.punctuation) | {'“', '—', '’', '”', '…'}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.wordnet = nltk.WordNetLemmatizer() 
    
    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories=categories)
        
        if fileids is None:
            return self.fileids() 
        
        return fileids
    
    def lemmatize(self, token, tag):
        token = token.lower()
        
        if token not in self.stopwords:
            if not all(c in self.punct for c in token):
                tag =  {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0], wn.NOUN)
                return self.wordnet.lemmatize(token, tag)
    
    def tokenize(self, doc):
        # Expects a preprocessed document, removes stopwords and punctuation
        # makes all tokens lowercase and lemmatizes them. 
        return list(filter(None, [
            self.lemmatize(token, tag)
            for paragraph in doc 
            for sentence in paragraph 
            for token, tag in sentence 
        ]))
    
    def docs(self, fileids=None, categories=None):
        # Resolve the fileids and the categories
        fileids = self._resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield self.tokenize(pickle.load(f))
    
    def labels(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        for fid in fileids:
            yield self.categories(fid)[0]


corpus = PickledCorpus('data/baleen_sample')


print("{} documents in {} categories".format(len(corpus.fileids()), len(corpus.categories())))


from nltk import ConditionalFreqDist

words = ConditionalFreqDist()

for doc, label in zip(corpus.docs(), corpus.labels()):
    for word in doc:
        words[label][word] += 1


for label, counts in words.items():
    print("{}: {:,} vocabulary and {:,} words".format(
        label, len(counts), sum(counts.values())
    ))


# ## Visualizing a Corpus 
# 
# TSNE - stochastic neighbor embedding, is a useful mechanism for performing high dimensional data visualization on text. We will use our classes to try to visualize groupings of documents on a per-class basis. 
# 

from sklearn.manifold import TSNE 
from sklearn.pipeline import Pipeline 
from sklearn.decomposition import TruncatedSVD 
from sklearn.feature_extraction.text import CountVectorizer 

cluster = Pipeline([
        ('vect', CountVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False)), 
        ('svd', TruncatedSVD(n_components=50)), 
        ('tsne', TSNE(n_components=2))
    ])

docs = cluster.fit_transform(list(corpus.docs()))


import seaborn as sns
import matplotlib.pyplot as plt 

from collections import defaultdict 

sns.set_style('whitegrid')
sns.set_context('notebook')

colors = {
    "design": "#e74c3c",
    "tech": "#3498db",
    "business": "#27ae60",
    "gaming": "#f1c40f",
    "politics": "#2c3e50",
    "news": "#bdc3c7",
    "cooking": "#d35400",
    "data_science": "#1abc9c",
    "sports": "#e67e22",
    "cinema": "#8e44ad",
    "books": "#c0392b",
    "do_it_yourself": "#34495e",
}

series = defaultdict(lambda: {'x':[], 'y':[]})
for idx, label in enumerate(corpus.labels()):
    x, y = docs[idx]
    series[label]['x'].append(x)
    series[label]['y'].append(y)

    
fig = plt.figure(figsize=(12,6))
ax = plt.subplot(111)
    
for label, points in series.items():
    ax.scatter(points['x'], points['y'], c=colors[label], alpha=0.7, label=label)

# Add a title 
plt.title("TSNE Projection of the Baleen Corpus")
    
# Remove the ticks 
plt.yticks([])
plt.xticks([])

# Add the legend 
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# ## Training a Model 
# 
# We'll build a model that can classify what hobby a document is about based on our sample corpus.
# 
# We'll need to add transformers that vectorize our text, and send them to a classification model. 
# 
# In this case we will evaluate with 12-part cross validation, using the `cross_val_predict` function and the `classifier_report` function.
# 
# The function `cross_val_predict` has a similar interface to `cross_val_score`, but returns, for each element in the input, the prediction that was obtained for that element when it was in the test set. Only cross-validation strategies that assign all elements to a test set exactly once can be used (otherwise, an exception is raised).
# 

hobbies = ['gaming', 'cooking', 'sports', 'cinema', 'books', 'do_it_yourself']

X = list(corpus.docs(categories=hobbies))
y = list(corpus.labels(categories=hobbies))


# Models 
from sklearn.linear_model import SGDClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier

# Transformers 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline 

# Evaluation 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

def identity(words): 
    return words 


# SVM Classifier 
svm = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('svm', SGDClassifier()), 
    ])

yhat = cross_val_predict(svm, X, y, cv=12)
print(classification_report(y, yhat))


# Logistic Regression 
logit = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('logit', LogisticRegression()), 
    ])

yhat = cross_val_predict(logit, X, y, cv=12)
print(classification_report(y, yhat))


# Naive Bayes
nbayes = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('nbayes', MultinomialNB()), 
    ])

yhat = cross_val_predict(nbayes, X, y, cv=12)
print(classification_report(y, yhat))


# Random Forest 
trees = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('trees', RandomForestClassifier()), 
    ])

yhat = cross_val_predict(trees, X, y, cv=12)
print(classification_report(y, yhat))


# ## Operationalization 
# 
# At this point we can save our best performing model to disk and use it to classify new text. 
# 
# The most important thing to remember is that the input to our model needs to be identical to the input we trained our model upon. Because we preprocessed our text in the experimental phase, we have to preprocess it before we make predictions on it as well. 
# 

def build_model(path, corpus):
    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)), 
        ('svm', SGDClassifier(loss='log')), 
    ])
    
    # Train model on the entire data set 
    X = list(corpus.docs(categories=hobbies))
    y = list(corpus.labels(categories=hobbies))
    model.fit(X, y)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)

build_model('data/hobbies.classifier', corpus)


# We can now load our model from disk 
with open('data/hobbies.classifier', 'rb') as f:
    model = pickle.load(f)


# Let's create a normalization method for fetching URL content
# that our model expects, based on our methods above. 
def fetch(url):
    html = requests.get(url)
    text = extract(html.text)
    tokens = tokenize(text)
    tags = tag(tokens)
    lemmas = lemmatize(tags)
    return list(normalize(lemmas))


def predict(url):
    text = fetch(url)
    probs = zip(model.classes_, model.predict_proba([text])[0])
    label = model.predict([text])[0]
    
    print("y={}".format(label))
    for cls, prob in sorted(probs, key=lambda x: x[1]):
        print("  {}: {:0.3f}".format(cls, prob))


predict("http://minimalistbaker.com/5-ingredient-white-chocolate-truffles/")


get_ipython().magic('matplotlib inline')


# # Visual Model Selection with Yellowbrick
# 
# In this tutorial, we are going to look at scores for a variety of [Scikit-Learn](http://scikit-learn.org) models and compare them using visual diagnostic tools from [Yellowbrick](http://www.scikit-yb.org) in order to select the best model for our data. 
# 
# 
# ## About Yellowbrick
# 
# Yellowbrick is a new Python library that extends the Scikit-Learn API to incorporate visualizations into the machine learning workflow.
# 
# The Yellowbrick library is a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. Yellowbrick extends the Scikit-Learn API with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of the Scikit-Learn Pipeline process, providing visual diagnostics throughout the transformation of high dimensional data.
# 
# To learn more about Yellowbrick, visit http://www.scikit-yb.org.
# 
# 
# ## About the Data
# 
# This tutorial uses a version of the mushroom data set from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/). Our objective is to predict if a mushroom is poisionous or edible based on its characteristics. 
# 
# The data include descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family.  Each species was identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended (this latter class was combined with the poisonous one).  
# 
# Our file, "agaricus-lepiota.txt," contains information for 3 nominally valued attributes and a target value from 8124 instances of mushrooms (4208 edible, 3916 poisonous). 
#         
# Let's load the data with Pandas.
# 

import os
import pandas as pd

names = [
    'class',
    'cap-shape',
    'cap-surface',
    'cap-color'
]

mushrooms = os.path.join('data','agaricus-lepiota.txt')
dataset   = pd.read_csv(mushrooms)
dataset.columns = names
dataset.head()


features = ['cap-shape', 'cap-surface', 'cap-color']
target   = ['class']

X = dataset[features]
y = dataset[target]


# ## Feature Extraction 
# 
# Our data, including the target, is categorical. We will need to change these values to numeric ones for machine learning. In order to extract this from the dataset, we'll have to use Scikit-Learn transformers to transform our input dataset into something that can be fit to a model. Luckily, Sckit-Learn does provide a transformer for converting categorical labels into numeric integers: [`sklearn.preprocessing.LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Unfortunately it can only transform a single vector at a time, so we'll have to adapt it in order to apply it to multiple columns.
# 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None. 
    """
    
    def __init__(self, columns=None):
        self.columns  = [col for col in columns] 
        self.encoders = None
    
    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode. 
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns 
        
        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns 
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame. 
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])
        
        return output


# ## Modeling and Evaluation
# 
# ### Common metrics for evaluating classifiers
# 
# **Precision** is the number of correct positive results divided by the number of all positive results (e.g. _How many of the mushrooms we predicted would be edible actually were?_).
# 
# **Recall** is the number of correct positive results divided by the number of positive results that should have been returned (e.g. _How many of the mushrooms that were poisonous did we accurately predict were poisonous?_).
# 
# The **F1 score** is a measure of a test's accuracy. It considers both the precision and the recall of the test to compute the score. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.
# 
#     precision = true positives / (true positives + false positives)
# 
#     recall = true positives / (false negatives + true positives)
# 
#     F1 score = 2 * ((precision * recall) / (precision + recall))
# 
# 
# Now we're ready to make some predictions!
# 
# Let's build a way to evaluate multiple estimators --  first using traditional numeric scores (which we'll later compare to some visual diagnostics from the Yellowbrick library).
# 

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


def model_selection(X, y, estimator):
    """
    Test various estimators.
    """ 
    y = LabelEncoder().fit_transform(y.values.ravel())
    model = Pipeline([
         ('label_encoding', EncodeCategorical(X.keys())), 
         ('one_hot_encoder', OneHotEncoder()), 
         ('estimator', estimator)
    ])

    # Instantiate the classification model and visualizer
    model.fit(X, y)  
    
    expected  = y
    predicted = model.predict(X)
    
    # Compute and return the F1 score (the harmonic mean of precision and recall)
    return (f1_score(expected, predicted))


# Try them all!
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier


model_selection(X, y, LinearSVC())


model_selection(X, y, NuSVC())


model_selection(X, y, SVC())


model_selection(X, y, SGDClassifier())


model_selection(X, y, KNeighborsClassifier())


model_selection(X, y, LogisticRegressionCV())


model_selection(X, y, LogisticRegression())


model_selection(X, y, BaggingClassifier())


model_selection(X, y, ExtraTreesClassifier())


model_selection(X, y, RandomForestClassifier())


# ### Preliminary Model Evaluation
# 
# Based on the results from the F1 scores above, which model is performing the best?

# ## Visual Model Evaluation
# 
# Now let's refactor our model evaluation function to use Yellowbrick's `ClassificationReport` class, a model visualizer that displays the precision, recall, and F1 scores. This visual model analysis tool integrates numerical scores as well color-coded heatmap in order to support easy interpretation and detection, particularly the nuances of Type I and Type II error, which are very relevant (lifesaving, even) to our use case!
# 
# 
# **Type I error** (or a **"false positive"**) is detecting an effect that is not present (e.g. determining a mushroom is poisonous when it is in fact edible).
# 
# **Type II error** (or a **"false negative"**) is failing to detect an effect that is present (e.g. believing a mushroom is edible when it is in fact poisonous).
# 

from sklearn.pipeline import Pipeline
from yellowbrick.classifier import ClassificationReport


def visual_model_selection(X, y, estimator):
    """
    Test various estimators.
    """ 
    y = LabelEncoder().fit_transform(y.values.ravel())
    model = Pipeline([
         ('label_encoding', EncodeCategorical(X.keys())), 
         ('one_hot_encoder', OneHotEncoder()), 
         ('estimator', estimator)
    ])

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(model, classes=['edible', 'poisonous'])
    visualizer.fit(X, y)  
    visualizer.score(X, y)
    visualizer.poof()  


visual_model_selection(X, y, LinearSVC())


visual_model_selection(X, y, NuSVC())


visual_model_selection(X, y, SVC())


visual_model_selection(X, y, SGDClassifier())


visual_model_selection(X, y, KNeighborsClassifier())


visual_model_selection(X, y, LogisticRegressionCV())


visual_model_selection(X, y, LogisticRegression())


visual_model_selection(X, y, BaggingClassifier())


visual_model_selection(X, y, ExtraTreesClassifier())


visual_model_selection(X, y, RandomForestClassifier())


# ## Reflection
# 
#  1. Which model seems best now? Why?
#  2. Which is most likely to save your life?
#  3. How is the visual model evaluation experience different from numeric model evaluation?

# # Model Data Generator 
# 
# This notebook is designed to generate fake data sets from an underlying model, injecting normally distributed randomness to describe discrepencies between observations. 
# 

get_ipython().magic('matplotlib notebook')


import os 
import random

import numpy as np 
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def add_collinear_features(n, X):
    """
    Add n collinear features to X
    """
    if not n:
        return X 

    for _ in range(n):
        # select random column 
        col = X[:,random.randint(0, X.shape[1]-1)]
        col = col * np.random.uniform(-10, 10, 1) + np.random.normal(0, 6, 1)
        col = col.reshape(col.shape[0], 1)
        
        X = np.append(X, col, 1)

    return X 

def generate_normal_model(n, coefs, bias=0.0, mean=0.0, scale=1.0, randscale=1.0, collinearity=0):
    """
    Generates a normally distributed model of n samples for the
    specified coeficients, can also include a bias parameter. 
    If the coefs is an integer, generates a random array of coefs
    whose length is the integer passed in. 
    
    mean and scale refer to the randomness of the X vectors. 
    randscale refers to the amount of randomness added to the target
    
    The model can also add n collinear features, determined by the 
    collinearity parameter. 
    
    This function returns a 2 dimensional X array, a target vector 
    Y and the coefs that may have been generated.
    
    Note that this returns standardized vectors by default.
    """
    if isinstance(coefs, int):
        coefs = np.random.uniform(-10.0, 10, coefs)
    
    # Create an (n, k) matrix of data 
    k = len(coefs)
    X = np.random.normal(mean, scale, (n, k))
    
    # Add collinear features 
    X = add_collinear_features(collinearity, X)
    coefs = np.append(coefs, np.random.uniform(-10.0, 10, collinearity))
    
    # Compute y and add normally distributed random error
    y = np.dot(X, coefs) + bias 
    y += np.random.normal(0.0, randscale, n)

    # Return the data sets 
    return X,y, coefs


def plot_model(n, coefs=1, bias=0.0, mean=0.0, scale=1.0, randscale=1.0, collinearity=0):
    """
    Draw a random model with specified parameters. 
    """
    if isinstance(coefs, int):
        if coefs > 2 or coefs < 1: 
            raise ValueError("can only plot 1 or 2 coefs")
    elif isinstance(coefs, list):
        if len(coefs) > 2 or len(coefs) < 1:
            raise ValueError("can only plot 1 or 2 coefs")
    else:
        raise TypeError("unknown coefs type.")
    
    # Generate the model 
    X, y, coefs = generate_normal_model(n, coefs, bias, mean, scale, randscale, collinearity)
    
    # Determine if 2D or 3D 
    fig = plt.figure()
    if len(coefs) == 2:
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(X[:,0], X[:,1], y)
        
        xm, ym = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max()), np.linspace(X[:,1].min(), X[:,1].max()))
        zm = coefs[0]*xm + coefs[1]*ym + bias 
        
        ax.plot_wireframe(xm, ym, zm, alpha=0.5)
    
    else:
        ax = fig.add_subplot(111)
    
        ax.scatter(X, y)
        Xm = np.linspace(X.min(), X.max()).reshape(X.shape[0], 1)
        ym = np.dot(Xm, coefs) + bias 
        ax.plot(Xm, ym)

    plt.show()

plot_model(50, 2, 3, 3, 1.0, 4.0)


def save_data(path, X, y, w, suffix=""):
    """
    Writes data out to a directory, specified by the path. 
    """
    data = os.path.join(path, "dataset{}.txt".format(suffix))
    target = os.path.join(path, "target{}.txt".format(suffix))
    coefs = os.path.join(path, "coefs{}.txt".format(suffix))
    
    np.savetxt(data, X)
    np.savetxt(target, y)
    np.savetxt(coefs, w)


X, y, w = generate_normal_model(10000, 18, bias=0.0, mean=6.3, scale=1.0, randscale=3.0, collinearity=0)
save_data("../data/generated", X, y, w)


X, y, w = generate_normal_model(10000, 10, bias=0.0, mean=6.3, scale=1.0, randscale=23.0, collinearity=8)
save_data("../data/generated", X, y, w, "-collin")


X, y, w = generate_normal_model(100, 1, bias=0.0, mean=6.3, scale=1.0, randscale=3.0, collinearity=0)
save_data("../data/generated", X, y, w, "-demo")


