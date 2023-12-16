from tkinter import *
import tkinter as tk
import tkinter as ttk
import csv
import customtkinter as ctk
import openai

# Data Structures
import numpy as np
import pandas as pd
import json
# Corpus Processing
import re
import nltk.corpus
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
# K-Means
from sklearn import cluster
# Visualization and Analysis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud
# Map Viz
import folium
import branca.colormap as cm
from branca.element import Figure
# Map Viz
import geopandas as gpd
TotalWidth = 1550
TotalHeight = 900
global anthem_entry_var
global dfs
# anthem_entry_var = StringVar()


def CompareAnthems():
    global dfs
    ctk.CTkCanvas(win)
    screen = Canvas(master=win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    compare_anthems = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white',
                                   corner_radius=0)
    compare_anthems.pack()

    info_listbox = None
    select_all_var = None
    all_countries = None
    search_entry = None

    # def on_select(event):
    #     selected_indices = info_listbox.curselection()
    #     selected_countries = [info_listbox.get(idx) for idx in selected_indices]
    #     print("Selected Countries:", selected_countries)

    def on_select(event):
        global dfs
        selected_indices = info_listbox.curselection()
        selected_countries = [info_listbox.get(idx) for idx in selected_indices]
        print("Selected Countries:", selected_countries)

        data = pd.read_csv('datasets/anthems.csv', encoding='utf-8')
        data.columns = map(str.lower, data.columns)

        continents = ['Europe', 'South_America', 'North_America','Asia','Africa']
        data = data.loc[data['continent'].isin(continents)]
        data = data.loc[data['country'].isin(selected_countries)]
        # first_row_index = data.index[0]
        # print(first_row_index)
        # rows_as_strings = [f"{index}: {row}" for index, row in data.head(6).iterrows()]

        print(data.head(6))
        corpus = data['anthem'].tolist()
        corpus[2][0:447]

        # removes a list of words (ie. stopwords) from a tokenized list.
        def removeWords(listOfTokens, listOfWords):
            return [token for token in listOfTokens if token not in listOfWords]

        # applies stemming to a list of tokenized words
        def applyStemming(listOfTokens, stemmer):
            return [stemmer.stem(token) for token in listOfTokens]

        # removes any words composed of less than 2 or more than 21 letters
        def twoLetters(listOfTokens):
            twoLetterWord = []
            for token in listOfTokens:
                if len(token) <= 2 or len(token) >= 21:
                    twoLetterWord.append(token)
            return twoLetterWord

        def processCorpus(corpus, language):
            stopwords = nltk.corpus.stopwords.words(language)
            param_stemmer = SnowballStemmer(language)
            # countries_list = [line.rstrip('\n') for line in open('lists/countrie.txt')]  # Load .txt file line by line
            # nationalities_list = [line.rstrip('\n') for line in
            #                       open('lists/nationalities.txt')]  # Load .txt file line by line
            # other_words = [line.rstrip('\n') for line in
            #                open('lists/stopwords_scrapmaker.txt')]  # Load .txt file line by line

            for document in corpus:
                index = corpus.index(document)
                corpus[index] = corpus[index].replace(u'\ufffd', '8')  # Replaces the ASCII ' ' symbol with '8'
                corpus[index] = corpus[index].replace(',', '')  # Removes commas
                corpus[index] = corpus[index].rstrip('\n')  # Removes line breaks
                corpus[index] = corpus[index].casefold()  # Makes all letters lowercase

                corpus[index] = re.sub('\W_', ' ', corpus[index])  # removes specials characters and leaves only words
                corpus[index] = re.sub("\S*\d\S*", " ", corpus[
                    index])  # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
                corpus[index] = re.sub("\S*@\S*\s?", " ", corpus[index])  # removes emails and mentions (words with @)
                corpus[index] = re.sub(r'http\S+', '', corpus[index])  # removes URLs with http
                corpus[index] = re.sub(r'www\S+', '', corpus[index])  # removes URLs with www

                listOfTokens = word_tokenize(corpus[index])
                twoLetterWord = twoLetters(listOfTokens)

                listOfTokens = removeWords(listOfTokens, stopwords)
                listOfTokens = removeWords(listOfTokens, twoLetterWord)
                # listOfTokens = removeWords(listOfTokens, countries_list)
                # listOfTokens = removeWords(listOfTokens, nationalities_list)
                # listOfTokens = removeWords(listOfTokens, other_words)
                #
                # listOfTokens = applyStemming(listOfTokens, param_stemmer)
                # listOfTokens = removeWords(listOfTokens, other_words)

                corpus[index] = " ".join(listOfTokens)
                corpus[index] = unidecode(corpus[index])

            return corpus

        language = 'english'
        corpus = processCorpus(corpus, language)
        corpus[2][0:460]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())

        final_df = tf_idf

        row_for_k=("{} rows".format(final_df.shape[0]))
        print(row_for_k)
        print(final_df.T.nlargest(5, 0))

        def run_KMeans(max_k, data):
            max_k += 1
            kmeans_results = dict()
            for k in range(2, max_k):
                kmeans = cluster.KMeans(n_clusters=k
                                        , init='k-means++'
                                        , n_init=10
                                        , tol=0.0001
                                        , random_state=1
                                        , algorithm='lloyd')

                kmeans_results.update({k: kmeans.fit(data)})

            return kmeans_results

        def printAvg(avg_dict):
            for avg in sorted(avg_dict.keys(), reverse=True):
                print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))

        def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
            fig, ax1 = plt.subplots(1)
            fig.set_size_inches(8, 6)
            ax1.set_xlim([-0.2, 1])
            ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

            ax1.axvline(x=silhouette_avg, color="red",
                        linestyle="--")  # The vertical line for average silhouette score of all the values
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')

            y_lower = 10
            sample_silhouette_values = silhouette_samples(df,
                                                          kmeans_labels)  # Compute the silhouette scores for each sample
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                  edgecolor=color, alpha=0.7)

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                         str(i))  # Label the silhouette plots with their cluster numbers at the middle
                y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
            plt.show()

        def silhouette(kmeans_dict, df, plot=False):
            df = df.to_numpy()
            avg_dict = dict()
            for n_clusters, kmeans in kmeans_dict.items():
                kmeans_labels = kmeans.predict(df)
                silhouette_avg = silhouette_score(df, kmeans_labels)  # Average Score for all Samples
                avg_dict.update({silhouette_avg: n_clusters})

                if (plot): plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)

        if(row_for_k<'8'):
            k=2
        else:
            k = 8
        kmeans_results = run_KMeans(k, final_df)

        def get_top_features_cluster(tf_idf_array, prediction, n_feats):
            labels = np.unique(prediction)
            global dfs
            dfs = []
            for label in labels:
                id_temp = np.where(prediction == label)  # indices for each cluster
                x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
                sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
                features = vectorizer.get_feature_names_out()
                best_features = [(features[i], x_means[i]) for i in sorted_means]
                df = pd.DataFrame(best_features, columns=['features', 'score'])
                dfs.append(df)
            return dfs

        def plotWords(dfs, n_feats):

            plt.figure(figsize=(8, 4))
            for i in range(0, len(dfs)):
                plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
                sns.barplot(x='score', y='features', orient='h', data=dfs[i][:n_feats])
                plt.tight_layout()
                plt.show()

        best_result = max(kmeans_results.keys())

        if(best_result >=5 ):
            best_result = 5

        kmeans = kmeans_results.get(best_result)

        final_df_array = final_df.to_numpy()
        prediction = kmeans.predict(final_df)
        n_feats = 20
        dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
        print(dfs)
        plotWords(dfs, 13)


        def centroidsDict(centroids, index):
            a = centroids.T[index].sort_values(ascending=False).reset_index().values
            centroid_dict = dict()

            for i in range(0, len(a)):
                centroid_dict.update({a[i, 0]: a[i, 1]})

            return centroid_dict

        def generateWordClouds(centroids):
            wordcloud = WordCloud(max_font_size=100, background_color='white')
            for i in range(0, len(centroids)):
                centroid_dict = centroidsDict(centroids, i)
                wordcloud.generate_from_frequencies(centroid_dict)

                plt.figure()
                plt.title('Cluster {}'.format(i))
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.show()

        centroids = pd.DataFrame(kmeans.cluster_centers_)
        centroids.columns = final_df.columns
        generateWordClouds(centroids)


    def toggle_select_all():
        # Get the current state of the "Select All" checkbox
        select_all_state = select_all_var.get()

        # If checked, select all countries; otherwise, deselect all
        if select_all_state:
            info_listbox.select_set(0, tk.END)
        else:
            info_listbox.selection_clear(0, tk.END)

    def country_list_from_csv(file_path):
        countries = []
        with open('datasets/anthems.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                countries.append(row['Country'])
        return countries

    def filter_countries(search_query):
        # Filter countries based on the search query
        filtered_countries = [country for country in all_countries if search_query.lower() in country.lower()]
        return filtered_countries

    def update_listbox(search_query=""):
        # Update the content of the listbox based on the search query
        info_listbox.delete(0, tk.END)  # Clear the current items

        filtered_countries = filter_countries(search_query)

        for country in filtered_countries:
            info_listbox.insert(tk.END, country)

    def search_entry_changed(event):
        # global search_entry  # Make search_entry global
        if search_entry:
            # Update the listbox based on the search entry content
            search_query = search_entry.get()
            update_listbox(search_query)

    f1 = ctk.CTkFrame(master=compare_anthems, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    f1.place(x=0, y=0)

    text_var = tk.StringVar(value="Comparison")

    l1 = ctk.CTkLabel(f1, textvariable=text_var, width=200, height=70, fg_color="#4473c5", corner_radius=15,
                      text_color='white', font=('italic', 22))
    l1.place(relx=0.5, rely=0.03, anchor='center')

    text_var1 = tk.StringVar(value="Select the countries you want to compare:")

    l2 = ctk.CTkLabel(f1, textvariable=text_var1, text_color="#4473c5", fg_color='white', font=('italic', 22))
    l2.place(relx=0.5, rely=0.2, anchor='center')

    # Entry for searching countries
    search_entry = Entry(f1, width=20, font=("Helvetica", 22))
    search_entry.place(relx=0.45, rely=0.3, anchor='center')
    search_entry.bind("<KeyRelease>", search_entry_changed)

    # Create a variable to track whether all countries are selected
    select_all_var = BooleanVar()
    select_all_var.set(False)

    # Checkbox for selecting all countries
    select_all_checkbox = ttk.Checkbutton(f1, text="Select All", variable=select_all_var, command=toggle_select_all,
                                          font=("Helvetica", 22), background='white', fg='black', bd=5)
    select_all_checkbox.pack(pady=10, side=tk.LEFT)
    select_all_checkbox.place(relx=0.58, rely=0.3, anchor='center')

    # Create info_listbox for displaying countries with rounded corners
    # l1= ctk.CTkLabel(f1, text='Select the anthems from following list:', fg_color='#4473c5',bg_color='white', text_color='white', corner_radius=5, font=("Helvetica", 22))
    # l1.place(relx=0.5, rely=0.1, anchor='center')
    info_listbox = Listbox(f1, width=30, height=5, bg="lightblue", selectmode=tk.MULTIPLE, bd=5, relief=tk.RIDGE,
                           borderwidth=2, font=("Helvetica", 22))
    info_listbox.place(relx=0.5, rely=0.45, anchor='center')

    scrollbar = tk.Scrollbar(info_listbox, orient=tk.VERTICAL, command=info_listbox.yview)
    scrollbar.place(relx=0.98, rely=0.5, anchor='center', relheight=1)

    # Link Scrollbar to Listbox
    info_listbox.config(yscrollcommand=scrollbar.set)

    # Get the list of countries from the CSV file
    all_countries = country_list_from_csv('datasets/anthems.csv')

    # Insert countries into the info_listbox
    for country in all_countries:
        info_listbox.insert(tk.END, country)

    compare_button = ctk.CTkButton(f1, fg_color='#4473c5', font=('Helvetica', 22), text='Compare',command=lambda: on_select(None),
                                   height=50, width=200,text_color='white', corner_radius=10)
    compare_button.place(relx=0.5, rely=0.6, anchor='center')

    text_var2 = tk.StringVar(value="Generate anthem with compared words:")
    l3 = ctk.CTkLabel(f1, textvariable=text_var2, text_color="#4473c5", fg_color='white', font=('italic', 22))
    l3.place(relx=0.5, rely=0.68, anchor='center')

    # Use CTkButton instead of tk Button with rounded corners
    custom_button = ctk.CTkButton(f1, text="Generate", fg_color='#4473c5', text_color='white', height=50, width=200,
                                  command=lambda:
                                  Generate(None,1),
                                  corner_radius=10, font=("Helvetica", 22))
    custom_button.place(relx=0.5, rely=0.76, anchor='center')

    bk_button = ctk.CTkButton(f1, fg_color='#4473c5', font=('Helvetica', 18), text='Back', text_color='white',
                              height=50, width=70, corner_radius=10, command=main)
    bk_button.place(relx=0.02, rely=0.022, anchor='center')


def Generate(my_entry,a):
    if(a==0):
        user_anthem = my_entry.get()
    # global dfs
    ctk.CTkCanvas(win)
    screen = Canvas(master=win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    generate = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    generate.pack()

    f1 = ctk.CTkFrame(master=generate, height=TotalHeight, width=TotalWidth, fg_color='#4473c5', corner_radius=0)
    f1.place(x=0, y=0)
    # print(dfs)


    openai.api_key = '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'

    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0,)
        return response.choices[0].message["content"]

    # print(anthem_entry_var)
    if(a==0):
        prompt = "freedom,bravery,bright future, hardwork ,devotion genrate national anthem lyrics 10 lines only "
    if(a==1):
        # you
        # are
        # a
        # professional
        # national
        # anthem
        # poet
        # everyone
        # approaches
        # you
        # for new anthems you anthems are very beautiful and based on client requirements your anthems are always based on 10 to 15 lines using the given words generate anthem in the described way always reply in the form of lyrics of an anthem
        prompt = (f"generate some lyrics of national anthem using only words with highest value from given data 10 lines each stanza of 3 lines\n{dfs} ")
    response = get_completion(prompt)
    print(response)

    text_var = tk.StringVar(value="Anthem")
    label = ctk.CTkLabel(f1, textvariable=text_var, width=200, height=70, text_color="#4473c5", corner_radius=15,
                         fg_color='white', font=('italic', 22))
    label.place(relx=0.5, rely=0.03, anchor='center')

    f2 = ctk.CTkFrame(master=f1, height=TotalHeight - TotalHeight / 3, width=TotalWidth - TotalWidth / 4,
                      fg_color='white', corner_radius=20)
    f2.place(relx=0.5, rely=0.5, anchor='center')
    l2 = ctk.CTkLabel(f2, text_color="#4473c5", corner_radius=15,
                         fg_color='white', font=('italic', 22))
    l2.place(relx=0.5, rely=0.5, anchor='center')
    l2.configure(text=response)

    # anthem_listbox = Listbox(f2, width=100, height=10, bg="white", selectmode=tk.MULTIPLE, bd=5, relief=tk.RIDGE,
    #                        borderwidth=2, font=("Helvetica", 22))
    # anthem_listbox.place(relx=0.5, rely=0.5, anchor='center')
    #
    # scrollbar = tk.Scrollbar(anthem_listbox, orient=tk.VERTICAL, command=anthem_listbox.yview)
    # scrollbar.place(relx=0.98, rely=0.5, anchor='center', relheight=1)
    #
    # anthem_listbox.config(yscrollcommand=scrollbar.set)
    #
    # anthem_listbox.insert(tk.END, response)

    bk_button = ctk.CTkButton(f1, text_color='#4473c5', font=('Helvetica', 18), text='Back', fg_color='white',
                              height=50, width=70, corner_radius=10, command=main)
    bk_button.place(relx=0.02, rely=0.02, anchor='center')





def GenerateAnthtems():
    ctk.CTkCanvas(win)
    screen = Canvas(master=win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    generate_anthems = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white',
                                    corner_radius=0)
    generate_anthems.pack()


    def input_words():
        input_words = my_entry.get()
        print(input_words)

    f1 = ctk.CTkFrame(master=generate_anthems, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    f1.place(x=0, y=0)

    text_var = tk.StringVar(value="Generation")

    label = ctk.CTkLabel(f1, textvariable=text_var, width=200, height=70, fg_color="#4473c5", corner_radius=15,
                         text_color='white', font=('italic', 22))
    label.place(relx=0.5, rely=0.03, anchor='center')

    f2 = ctk.CTkFrame(f1, height=400, width=400, fg_color='#4473c5', corner_radius=20)
    f2.place(relx=0.5, rely=0.48, anchor='center')

    text_var1 = tk.StringVar(value="Enter the words with which you want to\n generate anthem :-)")

    label = ctk.CTkLabel(master=f2, textvariable=text_var1, height=70, text_color='#4473c5', font=('italic', 18),
                         fg_color='white', corner_radius=10)
    label.place(relx=0.5, rely=0.2, anchor='center')


    my_entry = ctk.CTkEntry(master=f2, fg_color='white',text_color='#4473c5', width=200, height=35, corner_radius=10)
    my_entry.place(relx=0.5, rely=0.5, anchor='center')
    # global var213123




    gnt_button = ctk.CTkButton(master=f2, fg_color='white', font=('italic', 16), text='Generate Anthem',
                               text_color='#4473c5', height=50, width=200, corner_radius=10, command=lambda: Generate(my_entry,0))
    gnt_button.place(relx=0.5, rely=0.8, anchor='center')

    bk_button = ctk.CTkButton(f1, fg_color='#4473c5', font=('Helvetica', 18), text='Back', text_color='white',
                              height=50, width=70, corner_radius=10, command=main)
    bk_button.place(relx=0.02, rely=0.02, anchor='center')

    # return var213123
def main():
    ctk.CTkCanvas(win)
    screen = Canvas(win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    main_page = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    main_page.pack()



    f1 = ctk.CTkFrame(master=main_page, height=TotalHeight, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f1.pack(side='right')
    f2 = ctk.CTkFrame(master=main_page, height=TotalHeight, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f2.pack(side='left')

    f3 = ctk.CTkFrame(master=f1, height=TotalHeight / 3, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f3.place(x=0, y=0)
    f4 = ctk.CTkFrame(master=f1, height=TotalHeight / 2, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f4.place(x=0, y=TotalHeight / 3)

    l1 = ctk.CTkLabel(f3, text='Anthem  Tinckler', fg_color='white', text_color='#4473c5', width=TotalWidth / 2,
                      font=('Helvetica', 60))
    l1.place(relx=0.45, rely=0.7, anchor='center')

    b1 = ctk.CTkButton(master=f4, text='Compare\n Anthems', fg_color='#4473c5', hover_color='#AAA9FF', width=200,
                       height=200, font=('Helvetica', 30), command=CompareAnthems)
    b1.place(relx=0.18, rely=0.3, anchor='w')
    b2 = ctk.CTkButton(master=f4, text='Generate\n Anthems', fg_color='#4473c5', hover_color='#AAA9FF', width=200,
                       height=200, font=('Helvetica', 30), command=GenerateAnthtems)
    b2.place(relx=0.82, rely=0.3, anchor='e')

    f5 = ctk.CTkFrame(master=f2, corner_radius=70, fg_color='#4473c5', height=TotalHeight, width=TotalWidth / 1.5)
    f5.place(x=-350, y=0)
    l2 = ctk.CTkLabel(f2,
                      text='A national anthem is not merely a musical composition; it becomes a sacred hymn, a poetic ode to the resilience and unity of a nation. It whispers the echoes of the past, sings the stories of its people, and orchestrates the dreams of a shared future, all harmonized in the language of music.',
                      fg_color='#4473c5', text_color='white', font=('Helvetica', 30), wraplength=TotalWidth / 4)
    l2.place(relx=0.45, rely=0.5, anchor='center')


win = ctk.CTk()
win.geometry("1550x1080")
win.title("National Anthem Analysis")
win.config(background='white')
# win.resizable(False,False)
main()
win.mainloop()