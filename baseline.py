from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import nltk
import statistics 
from time import time
from sklearn.cluster import KMeans
from nltk import word_tokenize
from random import shuffle
from newspaper import Article
import math



multAcc = []
multPre = []
multPreNo = []
multRe = []
multReNo = []
multF = []
multFNo = []
multTime = []

lrAcc = []
lrPre = []
lrRe = []
lrF = []
lrTime = []
lrPreNo = []
lrReNo = []
lrFNo = []


testAcc = []
testPre = []
testRe = []
testF = []
testTime = []
testPreNo = []
testReNo = []
testFNo = []

percepAcc = []
percepPre = []
percepRe = []
percepF = []
percepTime = []
percepPreNo = []
percepReNo = []
percepFNo = []


berAcc = []
berPre = []
berRe = []
berF = []
berTime = []
berPreNo = []
berReNo = []
berFNo = []

stopWords = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or', ',', '.', '-',\
                           '"', "”", "—", ":", ";", "(", ")"
                           ]


politicalArticles = ["http://www.nytimes.com/2015/12/04/us/politics/donald-trump-and-ben-carson-face-a-foreign-policy-test-before-a-republican-jewish-group.html?ref=politics",\
 					"http://www.nytimes.com/2015/12/04/world/middleeast/to-crush-isis-john-kerry-urges-deft-removal-of-syrias-assad.html?ref=politics", \
 					"http://www.cnn.com/2015/12/03/politics/gun-control-obamacare-planned-parenthood-senate/index.html", "http://www.cnn.com/2015/12/04/politics/hillary-clinton-republicans-head-to-head-poll/index.html", 
					"http://www.cnn.com/2015/12/04/politics/jeb-bush-campaign-free-fall-3-percent/index.html", "http://blogs.wsj.com/law/2015/12/04/obama-administration-defends-syrian-resettlement-plan-in-court/",\
				 	"http://blogs.wsj.com/washwire/2015/12/03/chris-christie-u-s-already-fighting-the-next-world-war/", "http://blogs.wsj.com/washwire/2015/12/03/time-is-running-short-for-congress-to-pass-tax-break-deal-this-year/"
					"http://www.nytimes.com/2015/12/05/us/politics/white-house-seeks-path-to-executive-action-on-gun-sales.html?ref=politics", \
					"http://www.nytimes.com/politics/first-draft/2015/12/04/g-o-p-field-continues-to-chase-donald-trump-new-poll-says/?ref=politics",\
					"https://www.washingtonpost.com/news/federal-eye/wp/2015/12/04/global-warming-sparks-partisan-firestorm-on-once-sleepy-house-committee/", \
					"http://www.nytimes.com/politics/first-draft/2015/12/04/renewal-of-export-import-bank-stings-conservative-republicans/?ref=politics", \
					"http://www.nytimes.com/politics/first-draft/2015/12/03/bush-super-pac-to-tell-jebs-story-in-15-minute-documentary/?ref=politics", \
					"http://www.nytimes.com/2015/12/04/world/middleeast/to-crush-isis-john-kerry-urges-deft-removal-of-syrias-assad.html?ref=politics", \
					"https://www.washingtonpost.com/news/post-politics/wp/2015/12/04/lindsey-grahams-moderate-primal-scream/",\
					"https://www.washingtonpost.com/news/post-politics/wp/2015/12/04/sanders-no-elected-official-should-be-shielded-in-wake-of-chicago-shooting/"
					"http://www.nytimes.com/2015/12/06/us/politics/95000-words-many-of-them-ominous-from-donald-trumps-tongue.html?ref=politics", \
					"http://www.nytimes.com/politics/first-draft/2015/12/04/a-few-house-democrats-to-attend-prayer-services-at-u-s-mosques/?ref=politics", \
					"http://www.nytimes.com/2015/12/04/us/politics/donald-trump-and-ben-carson-face-a-foreign-policy-test-before-a-republican-jewish-group.html?ref=politics", \
					"http://www.reuters.com/article/us-california-shooting-obama-idUSKBN0TO0B520151205", \
					"http://www.reuters.com/article/us-usa-court-puertorico-idUSKBN0TN2BH20151205", "http://www.reuters.com/article/us-usa-congress-transportation-whitehous-idUSKBN0TN29M20151204", \
					"http://www.reuters.com/article/us-usa-election-debate-idUSKCN0SR00K20151102", "http://www.reuters.com/article/us-mideast-crisis-usa-republicans-idUSKCN0SQ2CA20151101", \
					"http://www.reuters.com/article/us-usa-election-bush-idUSKCN0R82TW20150909", "http://www.reuters.com/article/us-usa-election-bush-idUSKCN0R82TW20150909", \
					"http://www.reuters.com/article/us-iran-nuclear-congress-idUSKCN0R81RB20150909", "http://www.reuters.com/article/us-usa-puertorico-healthcare-idUSKCN0R82JB20150908", \
					"http://www.reuters.com/article/us-iran-nuclear-congress-wyden-idUSKCN0R81Z020150908", "http://www.reuters.com/article/us-altegrity-lawsuit-settlement-idUSKCN0QO2AQ20150819", \
					"http://www.reuters.com/article/us-sanctions-mexico-idUSKCN0QO20E20150819", "http://www.reuters.com/article/us-usa-microbeads-wisconsin-idUSKCN0PC01B20150702",\
					"http://www.reuters.com/article/us-usa-california-racketeering-idUSKCN0PB5TA20150701","http://www.npr.org/2015/12/04/458334512/house-committee-report-finds-secret-service-is-an-agency-in-crisis",\
					"http://www.nytimes.com/politics/first-draft/2015/12/05/at-donald-trump-rally-in-north-carolina-the-protesters-just-keep-coming/?ref=politics", "http://www.nytimes.com/2015/12/05/world/americas/cuba-and-us-to-discuss-settling-claims-on-property.html?ref=politics",\
					"http://www.npr.org/2015/12/03/458304236/senate-expected-to-pass-bill-to-defund-planned-parenthood-repeal-health-law","http://www.npr.org/2015/12/03/458309968/chicago-mayor-emanuel-now-says-he-welcomes-federal-investigation",\
					"http://www.npr.org/2015/12/01/458087304/republican-candidates-slam-obamas-focus-on-climate-change","http://www.npr.org/2015/12/01/458059894/in-wake-of-paris-attacks-chris-christie-sees-an-opportunity",\
					"http://www.npr.org/sections/thetwo-way/2015/12/01/458060154/congress-strikes-deal-on-5-year-transportation-bill","http://www.npr.org/2015/12/01/457902166/iowa-evangelicals-warm-to-ted-cruz",\
					"http://www.npr.org/2015/11/30/457904555/obamas-final-state-of-the-union-set-for-jan-12","http://www.npr.org/2015/11/19/456683695/sanders-speech-highlights-generational-divide-over-socialism",\
					"http://www.npr.org/2015/11/19/456683660/hillary-clinton-calls-for-shift-in-strategy-to-destroy-isis", "http://www.npr.org/sections/thetwo-way/2015/11/19/456651251/house-votes-to-increase-security-checks-on-refugees-from-iraq-syria",\
					"http://www.npr.org/2015/11/19/456619737/jeb-bush-proves-money-isnt-everything-in-politics","http://www.npr.org/2015/11/19/456560662/superpac-or-not-this-group-has-money-to-bern-for-sanders", \
					"http://www.npr.org/2015/11/18/456549211/elizabeth-warren-steps-up-campaign-for-liberal-agenda","http://www.npr.org/2015/11/17/456435101/obama-administration-briefs-governors-on-refugee-screening", \
					"http://www.npr.org/2015/11/17/456432166/supreme-court-asked-to-take-messy-case-on-interstate-same-sex-adoption", "http://www.npr.org/2015/11/17/456416876/jindal-ends-presidential-campaign-this-is-not-my-time"]

notPoliticalArticles = ["http://www.nytimes.com/2015/12/05/business/media/the-news-medias-grim-playbook-on-mass-shootings.html",\
						"http://www.nytimes.com/2015/12/04/business/starbucks-prospers-by-keeping-pace-with-the-coffee-snobs.html?src=me",\
						"http://www.nytimes.com/2015/12/04/technology/zuckerberg-explains-the-details-of-his-philanthropy.html?ref=technology", \
						"http://www.nytimes.com/2015/12/06/nyregion/in-the-spirit-of-mark-zuckerberg-an-experimental-school-in-brooklyn.html?ref=technology",\
						"http://www.nytimes.com/2015/12/06/realestate/new-apps-and-services-for-renters.html?ref=technology",
						"http://www.nytimes.com/2015/12/04/science/crispr-cas9-human-genome-editing-moratorium.html?ref=health", "http://www.nytimes.com/2015/12/08/science/parents-may-pass-down-more-than-just-genes-study-suggests.html?ref=health", \
						"http://www.nytimes.com/2015/12/05/sports/soccer/fifas-message-is-at-the-mercy-of-the-wrong-messengers.html?ref=sports", "http://www.cnn.com/2015/12/04/entertainment/adele-new-york-city-concert-feat/index.html",\
						"http://www.cnn.com/2015/12/01/entertainment/red-charity-world-aids-day-bono-feat/index.html"
						"https://www.washingtonpost.com/news/arts-and-entertainment/wp/2015/11/30/gates-rockefeller-rubenstein-discuss-life-and-philanthropy-at-smithsonian-museum/", \
						"https://www.washingtonpost.com/news/arts-and-entertainment/wp/2015/12/04/teyonah-parris-is-the-new-face-of-beauty-in-spike-lees-chi-raq/", \
						"http://www.cnn.com/2015/11/03/health/gupta-sugar-study-kids/index.html", \
						"http://www.cnn.com/2015/12/01/opinions/sex-trafficking-tenancingo-polaris/index.html", "http://money.cnn.com/2015/12/03/technology/uber-rivals-lyft-didi-kuadi/index.html", \
						"http://money.cnn.com/2015/11/30/technology/iphone-7-headphone-jack/index.html"
						"http://www.nytimes.com/2015/12/06/nyregion/a-renowned-dance-couple-keeps-flamencos-flame-burning-in-new-york.html?ref=nyregion&_r=0", \
						"http://www.nytimes.com/2015/12/06/nyregion/a-chelsea-vintage-shop-with-an-undying-love-for-ornaments.html?ref=nyregion", \
						"http://www.nytimes.com/2015/12/06/nyregion/how-jamie-hodari-workplace-entrepreneur-spends-his-sundays.html?ribbon-ad-idx=5&rref=nyregion&module=Ribbon&version=origin&region=Header&action=click&contentCollection=New%20York&pgtype=article", \
						"http://www.nytimes.com/2015/12/09/dining/hungry-city-kopitiam-lower-east-side.html?ref=dining", "http://www.cnn.com/2015/04/09/living/cnnheroes-doyne/index.html", \
						"http://money.cnn.com/2015/12/04/news/australia-beer-recall-broken-glass/index.html?iid=ob_homepage_deskrecommended_pool&iid=obnetwork", \
						"http://money.cnn.com/2015/12/03/news/hoverboards-fire-recall-uk/index.html?iid=ob_homepage_tech_pool_mobile&iid=obnetwork", "http://www.reuters.com/article/us-vtech-cyberattack-kids-analysis-idUSKBN0TP0FQ20151206", \
						"http://www.reuters.com/article/us-california-shooting-socialmedia-insig-idUSKBN0TO0OS20151206", "http://www.reuters.com/article/us-apple-samsung-payment-idUSKBN0TN20R20151204", \
						"http://www.reuters.com/article/us-yahoo-divestiture-idUSKBN0TN2IC20151205", "http://www.reuters.com/article/us-amazon-bookstore-idUSKCN0SS08B20151103", \
						"http://www.reuters.com/article/us-privateequity-lonsdale-lawsuit-idUSKCN0SR23Q20151103", "http://www.reuters.com/article/us-china-counterfeits-idUSKCN0SS02820151103", \
						"http://www.reuters.com/article/us-usa-drones-alphabet-idUSKCN0SR20520151103", "http://www.reuters.com/article/us-usa-stocks-weekahead-idUSKBN0TN2KU20151205", \
						"http://www.reuters.com/article/us-gm-china-usa-idUSKBN0TN2GP20151204", "http://www.reuters.com/article/us-e-i-du-pont-de-hedgefunds-trian-idUSKCN0S031220151006", \
						"http://www.reuters.com/article/us-bombardier-airbus-statement-idUSKCN0S031E20151006", "http://www.reuters.com/article/us-volkswagen-emissions-mueller-idUSKCN0S02XH20151007", \
						"http://www.reuters.com/article/us-markets-oil-idUSKCN0S002Q20151006","http://www.npr.org/sections/thetwo-way/2015/12/04/458502872/frenzied-media-pour-over-home-of-san-bernardino-killers-during-live-broadcasts",\
						"http://www.npr.org/sections/parallels/2015/12/03/458353036/as-saudi-arabia-battles-its-oil-rivals-prices-are-expected-to-stay-low", \
						"http://www.npr.org/sections/thetwo-way/2015/12/04/458351219/effort-to-build-the-worlds-biggest-telescope-hits-a-big-snag", "http://www.npr.org/sections/thetwo-way/2015/12/03/456533421/former-coal-ceo-blankenship-found-guilty-of-conspiracy-in-mine-disaster-case", \
						"http://www.npr.org/sections/health-shots/2015/12/03/458216778/specialty-drugs-can-prove-expensive-even-with-medicare-coverage", "http://www.npr.org/sections/thetwo-way/2015/11/30/457900449/bill-gates-and-other-billionaires-pledge-to-take-on-climate-change", \
						"http://www.npr.org/sections/thetwo-way/2015/11/30/457402547/why-negotiators-at-paris-climate-talks-are-tossing-the-kyoto-model", "http://www.npr.org/sections/alltechconsidered/2015/12/06/458347976/you-can-give-a-robot-a-paintbrush-but-does-it-create-art", \
						"http://www.npr.org/2015/11/27/457555217/holiday-shoppers-get-a-head-start-on-thanksgiving", "http://www.npr.org/sections/alltechconsidered/2015/11/25/457255846/from-takeout-to-breakups-apps-can-deliver-anything-for-a-price", \
						"http://www.npr.org/sections/thesalt/2015/11/24/457247226/cranberry-you-could-eat-without-sugar", "http://www.npr.org/sections/thetwo-way/2015/11/23/457090827/in-wake-of-attacks-france-moves-to-regulate-prepaid-bank-cards", \
						"http://www.npr.org/2015/11/21/456893868/ford-workers-approve-contract-with-uaw-by-slim-margin"]

print(len(politicalArticles))
print(len(notPoliticalArticles))
allArticles = []
allArticlesBinary = []




for url in politicalArticles:
	polart = Article(url, language='en')
	polart.download()
	polart.parse()
	wordsSeen = set()
	thisArticle = dict()
	thisArtBinary = dict()
	tokens = word_tokenize(polart.text)
	for token in tokens:
		token = token.lower()
		if token not in wordsSeen and token not in stopWords:
			thisArticle[token] = 1
			thisArtBinary[token] = True
			wordsSeen.add(token)
		elif token not in stopWords and token in wordsSeen:
				
				thisArticle[token] += 1
	tup = (thisArticle, 'political')
	tup2 = (thisArtBinary, 'political')
	allArticles.append(tup)
	allArticlesBinary.append(tup2)

for url in notPoliticalArticles:
	noPolart = Article(url)
	noPolart.download()
	noPolart.parse()
	wordsSeen = set()
	thisArticle = dict()
	thisArtBinary = dict()
	tokens = word_tokenize(noPolart.text)
	for token in tokens:
		token = token.lower()
		if token not in wordsSeen and token not in stopWords:
			thisArticle[token] = 1
			thisArtBinary[token] = True
			wordsSeen.add(token)
		elif token not in stopWords and token in wordsSeen:
			thisArticle[token] += 1
	tup = (thisArticle, 'notPolitical')
	tup2 = (thisArtBinary, 'notPolitical')
	allArticles.append(tup)
	allArticlesBinary.append(tup2)


def calcPRF(name, answers, returned):
	pre = []
	preNo = []
	re = []
	reNo = []
	fsc = []
	fscNo = []
	report = precision_recall_fscore_support(answers,returned)
	if report[0][0] != 0. and report[0][0]!=0.0:
		pre.append(report[0][0])
	if report[0][1] != 0. and report[0][1] != 0.0:
		preNo.append(report[0][1]) 
	if report[1][0] != 0. and report[1][0] != 0.0:
		re.append(report[1][0])
	if report[1][1] != 0. and report[1][1] != 0.0:
		reNo.append(report[1][1])
	if report[2][0] != 0. and report[2][0] != 0.0:
		fsc.append(report[2][0]) 
	if report[2][1] != 0. and report[2][1] != 0.0:
		fscNo.append(report[2][1])
	if name == 'mult':
		for it in pre:
			multPre.append(it)
		for it in re:
			multRe.append(it)
		for it in fsc:
			multF.append(it)
		for it in preNo:
			multPreNo.append(it)
		for it in reNo:
			multReNo.append(it)
		for it in fscNo:
			multFNo.append(it)
	elif name == 'lr':
		for it in pre:
			lrPre.append(it)
		for it in re:
			lrRe.append(it)
		for it in fsc:
			lrF.append(it)
		for it in preNo:
			lrPreNo.append(it)
		for it in reNo:
			lrReNo.append(it)
		for it in fscNo:
			lrFNo.append(it)
	elif name == 'test':
		for it in pre:
			testPre.append(it)
		for it in re:
			testRe.append(it)
		for it in fsc:
			testF.append(it)
		for it in preNo:
			testPreNo.append(it)
		for it in reNo:
			testReNo.append(it)
		for it in fscNo:
			testFNo.append(it)
	elif name == 'percep':
		for it in pre:
			percepPre.append(it)
		for it in re:
			percepRe.append(it)
		for it in fsc:
			percepF.append(it)
		for it in preNo:
			percepPreNo.append(it)
		for it in reNo:
			percepReNo.append(it)
		for it in fscNo:
			percepFNo.append(it)
	elif name == 'ber':
		for it in pre:
			berPre.append(it)
		for it in re:
			berRe.append(it)
		for it in fsc:
			berF.append(it)
		for it in preNo:
			berPreNo.append(it)
		for it in reNo:
			berReNo.append(it)
		for it in fscNo:
			berFNo.append(it)

def run_stats():
	#shuffle articles
	shuffle(allArticles)
	shuffle(allArticlesBinary)
	numArticles = len(allArticles)
	split = math.floor(numArticles * .75)
	trainingSet = allArticles[:split]
	test = allArticles[split:]
	bitrain = allArticlesBinary[:split]
	bitest = allArticlesBinary[split:]
	testSet = []
	bitestSet = []
	testAnswers = []
	for item in test:
		testSet.append(item[0])
		testAnswers.append(item[1])
	for item in bitest:
		bitestSet.append(item[0])


	multClassif = SklearnClassifier(MultinomialNB())
	ti = time()
	multClassif.train(trainingSet)
	multRes = multClassif.classify_many(testSet)
	t0 = time() - ti
	multTime.append(t0)
	multAcc.append(accuracy_score(testAnswers, multRes))
	calcPRF('mult', testAnswers, multRes)


	lrmult = SklearnClassifier(LogisticRegression())
	ti = time()
	lrmult.train(trainingSet)
	logRes = lrmult.classify_many(testSet)
	t3 = time() - ti
	lrTime.append(t3)
	lrAcc.append(accuracy_score(testAnswers, logRes))
	calcPRF('lr', testAnswers, logRes)

	pipe = Pipeline([('tfidf', TfidfTransformer()), 
					#('chi2', SelectKBest(chi2, k=500)),
					('nb', MultinomialNB())])
	testClassif = SklearnClassifier(pipe)
	ti = time()
	testClassif.train(trainingSet)
	testres = testClassif.classify_many(testSet)
	t5 = time() - ti
	testTime.append(t5)
	testAcc.append(accuracy_score(testAnswers, testres))
	calcPRF('test', testAnswers, testres)


	percepclass = SklearnClassifier(Perceptron())
	ti = time()
	percepclass.train(trainingSet)
	precepres = percepclass.classify_many(testSet)
	t7 = time() - ti
	percepTime.append(t7)
	percepAcc.append(accuracy_score(testAnswers, precepres))
	calcPRF('percep', testAnswers, precepres)


	berclass = SklearnClassifier(BernoulliNB())
	ti = time()
	berclass.train(bitrain)
	berres = berclass.classify_many(bitestSet)
	t9 = time() - ti
	berTime.append(t9)
	berAcc.append(accuracy_score(testAnswers, berres))
	calcPRF('ber', testAnswers, berres)


i = 0

while(i < 101):
	run_stats()
	i += 1

#print results
print("--------Naive Bayes Classifiers---------")
print("MultinomialNB")
print("Accuracy:  ", (sum(multAcc)/float(len(multAcc))), "	Time: ","   Avg time: ", sum(multTime)/float(len(multTime)) )
print("politics pre : ", (sum(multPre)/float(len(multPre))), "	no politics pre: ", (sum(multPreNo)/float(len(multPreNo))))
print("politics re: ",(sum(multRe)/float(len(multRe))), "	no politics re: ",(sum(multReNo)/float(len(multReNo)))) 
print("politics f: ", (sum(multF)/float(len(multF))), "	no politics f: ", (sum(multFNo)/float(len(multFNo))) ) 

print("BernoulliNB")
print("Accuracy:  ", (sum(berAcc)/float(len(berAcc))), "	Time: ","   Avg time: ", sum(berTime)/float(len(berTime)) )
print("politics pre : ", (sum(berPre)/float(len(berPre))), "	no politics pre: ", (sum(berPreNo)/float(len(berPreNo))))
print("politics re: ",(sum(berRe)/float(len(berRe))), "	no politics re: ",(sum(berReNo)/float(len(berReNo)))) 
print("politics f: ", (sum(berF)/float(len(berF))), "	no politics f: ", (sum(berFNo)/float(len(berFNo))) ) 

print("MultinomialNB w/ tf-idf")
print("Accuracy:  ", (sum(testAcc)/float(len(testAcc))), "	Time: ","   Avg time: ", sum(testTime)/float(len(testTime)) )
print("politics pre : ", (sum(testPre)/float(len(testPre))), "	no politics pre: ", (sum(testPreNo)/float(len(testPreNo))))
print("politics re: ",(sum(testRe)/float(len(testRe))), "	no politics re: ",(sum(testReNo)/float(len(testReNo)))) 
print("politics f: ", (sum(testF)/float(len(testF))), "	no politics f: ", (sum(testFNo)/float(len(testFNo))) ) 

print("--------Linear Model Classifiers--------")
print("Logisitic Regression")
print("Accuracy:  ", (sum(lrAcc)/float(len(lrAcc))), "	Time: ","   Avg time: ", sum(lrTime)/float(len(lrTime)) )
print("politics pre : ", (sum(lrPre)/float(len(lrPre))), "	no politics pre: ", (sum(lrPreNo)/float(len(lrPreNo))))
print("politics re: ",(sum(lrRe)/float(len(lrRe))), "	no politics re: ",(sum(lrReNo)/float(len(lrReNo)))) 
print("politics f: ", (sum(lrF)/float(len(lrF))), "	no politics f: ", (sum(lrFNo)/float(len(lrFNo))) ) 

print("Perceptron")
print("Accuracy:  ", (sum(percepAcc)/float(len(percepAcc))), "	Time: ","   Avg time: ", sum(percepTime)/float(len(percepTime)) )
print("politics pre : ", (sum(percepPre)/float(len(percepPre))), "	no politics pre: ", (sum(percepPreNo)/float(len(percepPreNo))))
print("politics re: ",(sum(percepRe)/float(len(percepRe))), "	no politics re: ",(sum(percepReNo)/float(len(percepReNo)))) 
print("politics f: ", (sum(percepF)/float(len(percepF))), "	no politics f: ", (sum(percepFNo)/float(len(percepFNo))) ) 


