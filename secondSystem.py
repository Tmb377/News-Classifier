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
multRe = []
multF = []
multTime = []
multPreO = []
multReO = []
multFO = []
multPreA = []
multReA = []
multFA = []
multPreS = []
multReS = []
multFS = []


lrAcc = []
lrPre = []
lrRe = []
lrF = []
lrTime = []
lrPreO = []
lrReO = []
lrFO = []
lrPreA = []
lrReA = []
lrFA = []
lrPreS = []
lrReS = []
lrFS = []


testAcc = []
testPre = []
testRe = []
testF = []
testTime = []
testPreO = []
testReO = []
testFO = []
testPreA = []
testReA = []
testFA = []
testPreS = []
testReS = []
testFS = []



percepAcc = []
percepPre = []
percepRe = []
percepF = []
percepTime = []
percepPreO = []
percepReO = []
percepFO = []
percepPreA = []
percepReA = []
percepFA = []
percepPreS = []
percepReS = []
percepFS = []



berAcc = []
berPre = []
berRe = []
berF = []
berTime = []
berPreO = []
berReO = []
berFO = []
berPreA = []
berReA = []
berFA = []
berPreS = []
berReS = []
berFS = []


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

sportsArticles = ["http://time.com/money/4137036/golden-state-warriors-ticket-prices-steph-curry/", "http://time.com/4132275/kobe-bryant-retire-2/", "http://time.com/4129805/expect-big-things-out-of-kobes-bryants-retirement/", \
				"http://time.com/4129125/kenya-iaaf-suspension/", "http://time.com/4128887/kobe-bryant-retirement-celebrities-reaction/", "http://time.com/4117684/soccer-turkey-fans-boo-silence-paris-attacks/", \
				"http://time.com/4116961/fanduel-draftkings-new-york-betting/", "http://time.com/4115735/new-orleans-saints-football-rob-ryan/", "http://time.com/4114013/ronda-rousey-holly-holm-upset/", \
				"http://www.nytimes.com/2015/12/08/sports/football/is-cam-newton-the-mvp-tom-brady-carson-palmer.html?ref=sports", "http://www.nytimes.com/2015/12/08/sports/football/jameis-winston-marcus-mariota-rookie-quarterbacks.html?ref=sports", \
				"http://www.nytimes.com/2015/12/08/sports/aroldis-chapman-traded-to-los-angeles-dodgers.html?ref=sports", "http://www.nytimes.com/2015/12/07/sports/football/moments-to-savor-for-2-jets-seeking-first-playoff-shot.html?ref=sports", \
				"http://www.nytimes.com/2015/12/07/sports/basketball/stephen-curry-golden-state-warriors-brooklyn-nets.html?ref=sports", "http://www.nytimes.com/2015/12/07/sports/soccer/portland-timbers-beat-columbus-crew-mls-cup.html?ref=sports", \
				"http://www.nytimes.com/2015/12/07/sports/baseball/new-york-mets-meetings-ben-zobrist-yoenis-cespedes.html?ref=sports", "http://www.nytimes.com/2015/12/07/sports/hockey/new-york-rangers-ottawa-senators-henrik-lundqvist.html?ref=sports", \
				"http://www.nytimes.com/2015/12/06/sports/baseball/ben-zobrist-far-from-a-star-is-now-coveted.html?ref=sports", "http://www.nytimes.com/2015/12/06/sports/hockey/jarome-iginla-is-chasing-a-milestone-with-no-shortage-of-force-or-speed.html?ref=sports", \
				"http://www.nytimes.com/2015/12/07/sports/skiing/lindsey-vonn-skiing-winter-sports-roundup.html?ref=sports", "http://www.nytimes.com/2015/12/07/sports/ncaafootball/college-football-playoff-michigan-state-clemson-rankings.html?ref=sports", \
				"http://www.nytimes.com/2015/12/07/sports/ncaafootball/final-four-clemson-oklahoma-michigan-state-alabama.html?ref=sports", \
				"http://www.reuters.com/article/us-nfl-vikings-idUSKBN0TQ01U20151207#TzOrgSLxx0iMX4gR.97", "http://www.reuters.com/article/us-motor-racing-renault-idUSKBN0TQ17H20151207#ou8dZFAGOQm8KTxd.97", \
				"http://www.reuters.com/article/us-nfl-playoffs-panthers-idUSKBN0TP0UW20151206#cMB250rIRyMD5DTh.97", "http://www.reuters.com/article/us-golf-challenge-idUSKBN0TP0WF20151207#KjtuSyggeioCbwQg.97", \
				"http://www.reuters.com/article/us-motor-racing-redbull-infiniti-idUSKBN0TP0YP20151207#jgkSt0lvy9v7llMD.97", "http://www.reuters.com/article/us-soccer-mls-idUSKBN0TP0XT20151207#P7y07DSsdAQLJH5H.97", \
				"http://www.reuters.com/article/us-nfl-browns-idUSKBN0TP0XB20151206#K4jrwozH2rU9gkj1.97", "http://www.reuters.com/article/us-alpineskiing-vonn-idUSKBN0TP0TW20151206#yo6agWRrIVIgTsud.97", \
				"http://www.reuters.com/article/us-soccer-china-idUSKBN0TP0JF20151206#s3Sy48iieK8jKHcC.97", "http://www.reuters.com/article/us-golf-asia-idUSKBN0TP0CZ20151206#hpUJSZY5sJymXyi5.97", \
				"http://www.reuters.com/article/us-baseball-redsox-price-idUSKBN0TN2NX20151204#3W47ATOFgvLWRHjw.97", "http://www.reuters.com/article/us-nfl-giants-burgess-idUSKBN0TN2E820151204#Vh9BtdyqdQB4cklg.97", \
				"http://www.reuters.com/article/us-baseball-mets-alderson-idUSKBN0TN2BJ20151204#dAZuediMfmAHOqBr.97", "http://www.reuters.com/article/us-baseball-marlins-bonds-idUSKBN0TN27R20151204#USHkAWckA4e0YBSV.97", \
				"http://www.reuters.com/article/us-safrica-pistorius-prosecutor-idUSKBN0TN26J20151204#OAFA64ESFYYHK3H0.97", "http://www.reuters.com/article/us-nba-knicks-porzingis-idUSKBN0TN22U20151204#woVVSqg6SetGYflw.97", \
				"http://www.reuters.com/article/us-nba-celtics-idUSKBN0TN0DK20151204#VPVctpSLhDdgAWPK.97"]

artEntArticles = ["http://www.reuters.com/article/us-music-adele-charts-idUSKBN0TQ2A320151207#EZCSueuiXjtrRDL4.97",\
					"http://www.reuters.com/article/us-film-starwars-secrecy-idUSKBN0TQ1I020151207#QpmXVClMXzlHIPdQ.97", \
					"http://www.reuters.com/article/us-france-shooting-u-idUSKBN0TQ02820151207#VVwA1q4Mg6jiuHei.97", \
					"http://www.reuters.com/article/us-swedish-crime-tv-idUSKBN0TQ1R620151207#ow5VVpcQ50rvWA4l.97",\
					"http://www.reuters.com/article/people-kardashian-baby-idUSKBN0TP01120151206#6yeWfszvQjuXLIiz.97",\
					"http://www.reuters.com/article/us-art-southeastasia-sotheby-s-idUSKBN0TM0AU20151203#aHxXbDJO7qxAdClS.97",\
					"http://www.reuters.com/article/us-netherlands-ukraine-stolen-art-idUSKBN0TQ1M220151207#LvuE56tcrrGWscqg.97",\
					"http://www.reuters.com/article/us-life-books-badsex-morrissey-idUSKBN0TK5OY20151203#e5g1wHFH5LfllMPG.97",\
					"http://www.reuters.com/article/us-japan-artist-mizuki-idUSKBN0TJ0KD20151130#VcAIOgXjPRrXPztG.97",\
					"http://www.reuters.com/article/us-theatre-awards-eveningstandard-idUSKBN0TB0Z920151123#QZqRczzabQce0XIj.97",\
					"http://www.npr.org/sections/thetwo-way/2015/12/01/458036861/heroic-women-in-strong-poses-serena-williams-amy-schumer-in-pirelli-calendar",\
					"http://www.npr.org/sections/codeswitch/2015/12/01/457944090/the-forgotten-actor-who-battled-hollywoods-whiteness-in-the-1950s",\
					"http://www.npr.org/sections/monkeysee/2015/11/30/457859235/a-charlie-brown-christmas-at-50",\
					"http://www.npr.org/sections/thetwo-way/2015/11/23/457125432/a-dress-judy-garland-wore-in-the-wizard-of-oz-sells-for-more-than-1-5-million",\
					"http://www.npr.org/2015/11/20/456812993/jessica-jones-struggles-in-life-but-triumphs-on-screen",\
					"http://www.npr.org/sections/thesalt/2015/11/09/455338415/salad-making-is-performance-art-at-the-getty-in-los-angeles",\
					"http://www.npr.org/sections/codeswitch/2015/11/03/453981918/stephen-park-on-playing-a-code-switching-character-in-steinbecks-east-of-eden",\
					"http://www.npr.org/2015/10/19/449890761/eddie-murphy-jokes-after-accepting-top-prize-for-humor",\
					"http://www.nytimes.com/2015/12/08/arts/music/grammy-awards-kendrick-lamar-taylor-swift-and-the-weeknd-lead-nominations.html?ref=arts",\
					"http://www.nytimes.com/2015/12/08/arts/design/art-basel-miami-beach-usually-sun-soaked-is-just-soaked-this-year.html?ref=arts",\
					"http://www.nytimes.com/2015/12/07/movies/holly-woodlawn-transgender-star-of-1970s-underground-films-dies-at-69.html?ref=arts",\
					"http://www.nytimes.com/2015/12/04/arts/design/prometheus-eternal-a-comic-book-anthology-in-the-company-of-masterpieces.html?ref=design",\
					"http://www.nytimes.com/2015/12/04/arts/design/wagner-collection-at-the-whitney-25-years-of-astute-buying.html?ribbon-ad-idx=17&rref=arts/design&module=Ribbon&version=context&region=Header&action=click&contentCollection=Art%20%26%20Design&pgtype=article",\
					"http://time.com/4138308/spotlight-mad-max-los-angeles-film-critics-association/",\
					"http://time.com/4138239/minecraft-nintendo-wii-u/", "http://time.com/4137890/the-leftovers-season-2-finale/", "http://time.com/4137889/taylor-swift-kangaroo-selfie/", \
					"http://time.com/4136064/the-wiz-nbc-review-live-musical/", "http://time.com/4135565/sisters-amy-poehler-tina-fey-star-wars-spoof/", "http://www.wsj.com/articles/how-the-trans-siberian-orchestra-became-a-holiday-hit-machine-1449170491", \
					"http://www.wsj.com/articles/giotto-litalia-review-the-artist-who-baptized-the-renaissance-1448390076", "http://time.com/4137774/ryan-gosling-snl-laughing-alien-abduction-video/", \
					"http://time.com/4124270/hey-arnold-new-tv-movie-nickelodeon-rugrats-doug-the-wild-thornberrys/", "http://www.huffingtonpost.com/entry/kris-jenner-60-swimsuit-love_5665ad34e4b08e945ff00dbd?utm_hp_ref=entertainment&ir=Entertainment&section=entertainment", \
					"http://www.huffingtonpost.com/entry/ryan-gosling-eva-mendes_56659fe4e4b079b2818f2573?utm_hp_ref=entertainment&ir=Entertainment&section=entertainment", \
					"http://www.huffingtonpost.com/entry/nene-leakes-the-view-shady_56658618e4b079b2818f1807?utm_hp_ref=entertainment&ir=Entertainment&section=entertainment", \
					"http://www.huffingtonpost.com/entry/happy-birthday-to-joan-didion-the-original-icon-of-impostor-syndrome_5661de8ae4b08e945fef54d4?utm_hp_ref=arts", \
					"http://www.huffingtonpost.com/entry/hans-scharer-erotic-paintings_565e29ebe4b072e9d1c3ed68?utm_hp_ref=arts"]

notPoliticalArticles = ["http://www.nytimes.com/2015/12/07/business/x-marks-the-spot-that-makes-online-ads-so-maddening.html?ref=technology", "http://www.nytimes.com/2015/12/08/upshot/why-its-too-soon-to-sour-on-the-zuckerberg-charity-plan.html?ref=technology", \
						"http://www.nytimes.com/2015/12/06/technology/personaltech/cant-put-down-your-device-thats-by-design.html?ref=technology", "http://www.nytimes.com/2015/12/07/opinion/eric-schmidt-on-how-to-build-a-better-web.html?ref=technology", \
						"http://bits.blogs.nytimes.com/2015/12/04/in-net-neutrality-hearing-judge-signals-comfort-with-f-c-c-s-defense/?ref=technology", "http://www.nytimes.com/2015/12/08/science/carbon-emissions-decline-peak-climate-change.html?rref=collection%2Fsectioncollection%2Fscience&action=click&contentCollection=science&region=rank&module=package&version=highlights&contentPlacement=1&pgtype=sectionfront", \
						"http://www.nytimes.com/2015/12/08/science/exercise-may-aid-brains-rewiring.html?rref=collection%2Fsectioncollection%2Fscience&action=click&contentCollection=science&region=stream&module=stream_unit&version=latest&contentPlacement=3&pgtype=sectionfront", \
						"http://www.nytimes.com/2015/12/08/health/hawaiis-dengue-fever-outbreak-grows.html?rref=collection%2Fsectioncollection%2Fscience&action=click&contentCollection=science&region=stream&module=stream_unit&version=latest&contentPlacement=4&pgtype=sectionfront", \
						"http://www.nytimes.com/2015/12/08/science/australias-feral-cats-most-likely-european.html?rref=collection%2Fsectioncollection%2Fscience&action=click&contentCollection=science&region=stream&module=stream_unit&version=latest&contentPlacement=7&pgtype=sectionfront", \
						"http://www.nytimes.com/2015/12/08/health/elder-caregivers-often-sacrifice-their-careers.html?rref=collection%2Fsectioncollection%2Fscience&action=click&contentCollection=science&region=stream&module=stream_unit&version=latest&contentPlacement=10&pgtype=sectionfront", \
						"http://www.nytimes.com/2015/12/08/business/merger-mania-in-hospitality-raises-competition-concerns.html", "http://www.nytimes.com/2015/12/08/business/dealbook/keurig-green-mountain-to-be-bought-for-13-9-billion.html?ref=business", \
						"http://www.nytimes.com/2015/12/08/business/energy-environment/despite-push-for-cleaner-cars-sheer-numbers-could-work-against-climate-benefits.html?ref=business", \
						"http://www.nytimes.com/2015/12/07/us/salaries-of-private-college-presidents-continue-to-rise-survey-finds.html?ref=business", "http://www.nytimes.com/2015/12/08/business/dealbook/ge-electrolux-appliances.html?ref=business", \
						"http://www.nytimes.com/2015/12/08/business/energy-environment/change-isnt-as-easy-as-a-flip-of-a-switch.html?ref=international", "http://www.nytimes.com/2015/12/04/business/starbucks-prospers-by-keeping-pace-with-the-coffee-snobs.html?ref=international", \
						"http://time.com/4138749/sufganiyot-jelly-doughnut-hanukkah-history/", "http://time.com/4127200/epa-founded-1970/", "http://time.com/4133522/chicken-tenders-recipe/", \
						"http://time.com/4130883/tv-sitting-cognitive-decline/", "http://time.com/4138476/aging-alzheimers-disease/", "http://time.com/4136368/phantom-menace-superbug-cdc/", \
						"http://time.com/4134758/anti-aging-science/", "http://time.com/4133353/pesticides-lung-function-children/", "http://time.com/4131551/eating-out-healthy/", "http://time.com/4134556/starbucks-panini-e-coli/", \
						"http://time.com/4133038/coffee-protect-type-2-diabetes/", "http://time.com/4130883/tv-sitting-cognitive-decline/", "http://time.com/4138454/spotify-year-in-music/", \
						"http://time.com/4139354/anonymous-isis-trolling-day/", "http://time.com/4136430/apple-spaceship-campus-drone/", "http://time.com/4132444/amazon-fire-tablet-blue-shade/", \
						"http://time.com/4130704/vtech-hack-childrens-toys/", "http://www.npr.org/sections/thesalt/2015/12/07/458490852/eat-up-how-cultural-messages-can-lead-to-eating-disorders", \
						"http://www.npr.org/sections/thesalt/2015/12/07/458063708/carbon-farming-gets-a-nod-at-paris-climate-conference", "http://www.npr.org/sections/thetwo-way/2015/12/07/458770044/beijing-issues-its-first-ever-red-alert-over-air-pollution", \
						"http://www.npr.org/sections/ed/2015/12/07/456208805/how-a-schools-attendance-number-hides-big-problems", "http://www.npr.org/sections/ed/2015/11/28/452898967/is-bigger-always-better-the-case-for-starting-small-with-new-learning-ideas", \
						"http://www.npr.org/sections/ed/2015/11/22/456779371/starting-a-high-school-from-scratch", "http://www.npr.org/sections/ed/2015/11/19/455378792/does-it-pay-to-pay-teachers-100-000", \
						"http://www.huffingtonpost.com/entry/jubilee-year-of-mercy_5661e72ce4b08e945fef5cd7?utm_hp_ref=religion", "http://www.huffingtonpost.com/entry/yoga-teacher-training-college_565f4266e4b08e945fedb502?utm_hp_ref=religion", \
						"http://www.huffingtonpost.com/entry/gratitude-practice-happiness_5655ec57e4b079b28189eab0?utm_hp_ref=religion", "http://www.huffingtonpost.com/entry/pope-francis-africa-trip_56562a99e4b072e9d1c1abfd?utm_hp_ref=religion", \
						"http://www.huffingtonpost.com/entry/pope-francis-world-war-3_5648ab7de4b06037734973e6?utm_hp_ref=religion"]

print("political ", len(politicalArticles))
print("everything ", len(notPoliticalArticles))
print("sports ", len(sportsArticles))
print("art ", len(artEntArticles))
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

for url in artEntArticles:
	artEntart = Article(url)
	artEntart.download()
	artEntart.parse()
	wordsSeen = set()
	thisArticle = dict()
	thisArtBinary = dict()
	tokens = word_tokenize(artEntart.text)
	for token in tokens:
		token = token.lower()
		if token not in wordsSeen and token not in stopWords:
			thisArticle[token] = 1
			thisArtBinary[token] = True
			wordsSeen.add(token)
		elif token not in stopWords and token in wordsSeen:
			thisArticle[token] += 1
	tup = (thisArticle, 'artEnt')
	tup2 = (thisArtBinary, 'artEnt')
	allArticles.append(tup)
	allArticlesBinary.append(tup2)

for url in sportsArticles:
	sportArt = Article(url)
	sportArt.download()
	sportArt.parse()
	wordsSeen = set()
	thisArticle = dict()
	thisArtBinary = dict()
	tokens = word_tokenize(sportArt.text)
	for token in tokens:
		token = token.lower()
		if token not in wordsSeen and token not in stopWords:
			thisArticle[token] = 1
			thisArtBinary[token] = True
			wordsSeen.add(token)
		elif token not in stopWords and token in wordsSeen:
			thisArticle[token] += 1
	tup = (thisArticle, 'sport')
	tup2 = (thisArtBinary, 'sport')
	allArticles.append(tup)
	allArticlesBinary.append(tup2)


def calcPRF(name, answers, returned):
	pre = []
	preO = []
	preA = []
	preS = []
	re = []
	reO = []
	reA = []
	reS = []
	fsc = []
	fscO = []
	fscA = []
	fscS = []
	report = precision_recall_fscore_support(answers,returned)
	if report[0][0] != 0. and report[0][0]!=0.0:
		pre.append(report[0][0])
	if report[0][1] != 0. and report[0][1] != 0.0:
		preO.append(report[0][1]) 
	if report[0][2] != 0. and report[0][2] != 0.0:
		preA.append(report[0][2])
	if report[0][3] != 0. and report[0][3] != 0.0:
		preS.append(report[0][3])
	if report[1][0] != 0. and report[1][0] != 0.0:
		re.append(report[1][0])
	if report[1][1] != 0. and report[1][1] != 0.0:
		reO.append(report[1][1])
	if report[1][2] != 0. and report[1][2] != 0.0:
		reA.append(report[1][2])
	if report[1][3] != 0. and report[1][3] != 0.0:
		reS.append(report[1][3])
	if report[2][0] != 0. and report[2][0] != 0.0:
		fsc.append(report[2][0]) 
	if report[2][1] != 0. and report[2][1] != 0.0:
		fscO.append(report[2][1])
	if report[2][2] != 0. and report[2][2] != 0.0:
		fscA.append(report[2][2])
	if report[2][3] != 0. and report[2][3] != 0.0:
		fscS.append(report[2][3])
	if name == 'mult':
		for it in pre:
			multPre.append(it)
		for it in re:
			multRe.append(it)
		for it in fsc:
			multF.append(it)
		for it in preO:
			multPreO.append(it)
		for it in reO:
			multReO.append(it)
		for it in fscO:
			multFO.append(it)
		for it in preA:
			multPreA.append(it)
		for it in reA:
			multReA.append(it)
		for it in fscA:
			multFA.append(it)
		for it in preS:
			multPreS.append(it)
		for it in reS:
			multReS.append(it)
		for it in fscS:
			multFS.append(it)

	elif name == 'lr':
		for it in pre:
			lrPre.append(it)
		for it in re:
			lrRe.append(it)
		for it in fsc:
			lrF.append(it)
		for it in preO:
			lrPreO.append(it)
		for it in reO:
			lrReO.append(it)
		for it in fscO:
			lrFO.append(it)
		for it in preA:
			lrPreA.append(it)
		for it in reA:
			lrReA.append(it)
		for it in fscA:
			lrFA.append(it)
		for it in preS:
			lrPreS.append(it)
		for it in reS:
			lrReS.append(it)
		for it in fscS:
			lrFS.append(it)

	elif name == 'test':
		for it in pre:
			testPre.append(it)
		for it in re:
			testRe.append(it)
		for it in fsc:
			testF.append(it)
		for it in preO:
			testPreO.append(it)
		for it in reO:
			testReO.append(it)
		for it in fscO:
			testFO.append(it)
		for it in preA:
			testPreA.append(it)
		for it in reA:
			testReA.append(it)
		for it in fscA:
			testFA.append(it)
		for it in preS:
			testPreS.append(it)
		for it in reS:
			testReS.append(it)
		for it in fscS:
			testFS.append(it)

	elif name == 'percep':
		for it in pre:
			percepPre.append(it)
		for it in re:
			percepRe.append(it)
		for it in fsc:
			percepF.append(it)
		for it in preO:
			percepPreO.append(it)
		for it in reO:
			percepReO.append(it)
		for it in fscO:
			percepFO.append(it)
		for it in preA:
			percepPreA.append(it)
		for it in reA:
			percepReA.append(it)
		for it in fscA:
			percepFA.append(it)
		for it in preS:
			percepPreS.append(it)
		for it in reS:
			percepReS.append(it)
		for it in fscS:
			percepFS.append(it)

	elif name == 'ber':
		for it in pre:
			berPre.append(it)
		for it in re:
			berRe.append(it)
		for it in fsc:
			berF.append(it)
		for it in preO:
			berPreO.append(it)
		for it in reO:
			berReO.append(it)
		for it in fscO:
			berFO.append(it)
		for it in preA:
			berPreA.append(it)
		for it in reA:
			berReA.append(it)
		for it in fscA:
			berFA.append(it)
		for it in preS:
			berPreS.append(it)
		for it in reS:
			berReS.append(it)
		for it in fscS:
			berFS.append(it)

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


print("BernoulliNB")
print("Accuracy: ", (sum(berAcc)/float(len(berAcc))))
print("Precision")
print("Politics: ",(sum(berPre)/float(len(berPre))) , "  Other: ",(sum(berPreO)/float(len(berPreO))) , "   Art: ",(sum(berPreA)/float(len(berPreA))) , "  Sports: ", (sum(berPreS)/float(len(berPreS))))
print("Recall")
print("Politics: ",(sum(berRe)/float(len(berRe))) , "  Other: ",(sum(berReO)/float(len(berReO))) , "   Art: ",(sum(berReA)/float(len(berReA))) , "  Sports: ", (sum(berReS)/float(len(berReS))))
print("F-Score")
print("Politics: ",(sum(berF)/float(len(berF))) , "  Other: ",(sum(berFO)/float(len(berFO))) , "   Art: ",(sum(berFA)/float(len(berFA))) , "  Sports: ", (sum(berFS)/float(len(berFS))))

print("MultinomialNB")
print("Accuracy: ", (sum(multAcc)/float(len(multAcc))))
print("Precision")
print("Politics: ",(sum(multPre)/float(len(multPre))) , "  Other: ",(sum(multPreO)/float(len(multPreO))) , "   Art: ",(sum(multPreA)/float(len(multPreA))) , "  Sports: ", (sum(multPreS)/float(len(multPreS))))
print("Recall")
print("Politics: ",(sum(multRe)/float(len(multRe))) , "  Other: ",(sum(multReO)/float(len(multReO))) , "   Art: ",(sum(multReA)/float(len(multReA))) , "  Sports: ", (sum(multReS)/float(len(multReS))))
print("F-Score")
print("Politics: ",(sum(multF)/float(len(multF))) , "  Other: ",(sum(multFO)/float(len(multFO))) , "   Art: ",(sum(multFA)/float(len(multFA))) , "  Sports: ", (sum(multFS)/float(len(multFS))))

print("MultinomialNB w/ tf-idf")
print("Accuracy: ", (sum(testAcc)/float(len(testAcc))))
print("Precision")
print("Politics: ",(sum(testPre)/float(len(testPre))) , "  Other: ",(sum(testPreO)/float(len(testPreO))) , "   Art: ",(sum(testPreA)/float(len(testPreA))) , "  Sports: ", (sum(testPreS)/float(len(testPreS))))
print("Recall")
print("Politics: ",(sum(testRe)/float(len(testRe))) , "  Other: ",(sum(testReO)/float(len(testReO))) , "   Art: ",(sum(testReA)/float(len(testReA))) , "  Sports: ", (sum(testReS)/float(len(testReS))))
print("F-Score")
print("Politics: ",(sum(testF)/float(len(testF))) , "  Other: ",(sum(testFO)/float(len(testFO))) , "   Art: ",(sum(testFA)/float(len(testFA))) , "  Sports: ", (sum(testFS)/float(len(testFS))))


print("--------Linear Model Classifiers--------")

print("Linear Regression")
print("Accuracy: ", (sum(lrAcc)/float(len(lrAcc))))
print("Precision")
print("Politics: ",(sum(lrPre)/float(len(lrPre))) , "  Other: ",(sum(lrPreO)/float(len(lrPreO))) , "   Art: ",(sum(lrPreA)/float(len(lrPreA))) , "  Sports: ", (sum(lrPreS)/float(len(lrPreS))))
print("Recall")
print("Politics: ",(sum(lrRe)/float(len(lrRe))) , "  Other: ",(sum(lrReO)/float(len(lrReO))) , "   Art: ",(sum(lrReA)/float(len(lrReA))) , "  Sports: ", (sum(lrReS)/float(len(lrReS))))
print("F-Score")
print("Politics: ",(sum(lrF)/float(len(lrF))) , "  Other: ",(sum(lrFO)/float(len(lrFO))) , "   Art: ",(sum(lrFA)/float(len(lrFA))) , "  Sports: ", (sum(lrFS)/float(len(lrFS))))

print("Perceptron")
print("Accuracy: ", (sum(percepAcc)/float(len(percepAcc))))
print("Precision")
print("Politics: ",(sum(percepPre)/float(len(percepPre))) , "  Other: ",(sum(percepPreO)/float(len(percepPreO))) , "   Art: ",(sum(percepPreA)/float(len(percepPreA))) , "  Sports: ", (sum(percepPreS)/float(len(percepPreS))))
print("Recall")
print("Politics: ",(sum(percepRe)/float(len(percepRe))) , "  Other: ",(sum(percepReO)/float(len(percepReO))) , "   Art: ",(sum(percepReA)/float(len(percepReA))) , "  Sports: ", (sum(percepReS)/float(len(percepReS))))
print("F-Score")
print("Politics: ",(sum(percepF)/float(len(percepF))) , "  Other: ",(sum(percepFO)/float(len(percepFO))) , "   Art: ",(sum(percepFA)/float(len(percepFA))) , "  Sports: ", (sum(percepFS)/float(len(percepFS))))


