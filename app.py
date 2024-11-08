import streamlit as st # for dl
import pandas as pd # for data frame
from langchain_huggingface import HuggingFaceEmbeddings # temp for now in the ht we will change to the IBM embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter # this is to sipletter fo the markdown df
from langchain_community.vectorstores import FAISS # VDB
from ibm_watsonx_ai.foundation_models import Model # IBM API call
import markdown # so the LLM can have getter understanding to the NHL

st.markdown(
    """
    <style>
    /* Hide the Streamlit top bar (header) */
    header { 
        visibility: hidden;
    }

    /* Set the background image */
    .stApp {
        background-image: url("https://raw.githubusercontent.com/salehtwj/Milsan/refs/heads/main/images/Screen Shot 1446-05-04 at 2.18.43 AM.png?raw=true");  
        background-size: cover;
        background-position: center;
    }

    /* Import IBM Plex Sans Arabic from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@100;200;300;400;500;600;700&display=swap');
    
    /* Set the direction of the entire app to right-to-left (RTL) */
    .stApp {
        direction: rtl;
    }

    /* Change font for the title */
    .stApp h1 {
        font-family: 'IBM Plex Sans Arabic', sans-serif;  /* Use IBM Plex Sans Arabic */
        font-size: 40px;
        text-align: right;  /* Align title to the right */
    }

    /* Change font for the text under the title (st.write) */
    .stApp p {
        font-family: 'IBM Plex Sans Arabic', sans-serif;  /* Use IBM Plex Sans Arabic */
        font-size: 18px;
        text-align: right;  /* Align text to the right */
    }
    

     /* Position the logos at the bottom left */
    .bottom-left-logos {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
    }

    .bottom-left-logos img {
        width: 100px;
        height: auto;
        margin-right: 15px;
    }


    /* Position the top left logo */
    .top-left-logo {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 9999;
    }

    .top-left-logo img {
        width: 100px;
        height: auto;
    }

    [data-baseweb="slider"] {
	direction: ltr;
    }
	[data-testid="stVerticalBlockBorderWrapper"]{

	background: rgba(255, 255, 255, 0.2);
	padding: 30px;
	border-radius: 15px;

    }

    [data-testid="stBaseButton-secondary"] {
    	background-color: rgb(38, 39, 48)
    }
    .custom-text, .custom-text p {
            font-family: 'IBM Plex Sans Arabic', sans-serif;
            font-size: 44px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="bottom-left-logos">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/SDAIA_logo-removebg-preview.png?raw=true" alt="SDAIA Logo">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/safcsp_logo.png?raw=true" alt="SAFCSP Logo">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/tuwaiq_logo_w.png?raw=true/" alt="Tuwaiq Logo">
    </div>

    <div class="top-left-logo">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/ALLaM_logo.png?raw=true" alt="ALLaM Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Load Data
data = {
    "Poem": [
         """قِفا نَبكِ مِن ذِكرى حَبيبٍ وَعِرفانِ  وَرَسمٍ عَفَت آياتُهُ مُنذُ أَزمانِ \n\n أَتَت حُجَجٌ بَعدي عَلَيها فَأَصبَحَت  كَخَطِّ زَبورٍ في مَصاحِفِ رُهبانِ"""
        ,""" أَعِنّي عَلى بَرقٍ أَراهُ وَميضِ  يُضيءُ حَبِيّاً في شَماريخَ بيضِ  \n\n وَيَهدَأُ تاراتٍ سَناهُ وَتارَةً  يَنوءُ كَتَعتابِ الكَسيرِ المَهيضِ  \n\n  وَتَخرُجُ مِنهُ لامِعاتٌ كَأَنَّها أَكُفٌّ تَلَقّى الفَوزَ عِندَ المَفيضِ """
        ,""" طَرِبتَ وَهاجَتكَ الظِباءُ السَوارِحُ غَداةَ غَدَت مِنها سَنيحٌ وَبارِحُ  \n\n  تَغالَت بِيَ الأَشواقُ حَتّى كَأَنَّما  بِزَندَينِ في جَوفي مِنَ الوَجدِ قادِحُ  \n\n وَقَد كُنتَ تُخفي حُبَّ سَمراءَ حِقبَةَ فَبُح لِانَ مِنها بِالَّذي أَنتَ بائِحُ """
        ,""" أُعاتِبُ دَهراً لا يَلينُ لِعاتِبِ وَأَطلُبُ أَمناً مِن صُروفِ النَوائِبِ \n\n وَتوعِدُني الأَيّامُ وَعداً تَغُرُّني وَأَعلَمُ حَقّاً أَنَّهُ وَعدُ كاذِبِ  \n\n خَدَمتُ أُناساً وَاِتَّخَذتُ أَقارِباً لِعَوني وَلَكِن أَصبَحوا كَالعَقارِبِ"""
        ,""" دَهَتني صُروفُ الدَهرِ وَاِنتَشَبَ الغَدرُ   وَمَن ذا الَّذي في الناسِ يَصفو لَهُ الدَهرُ  \n\n  وَكَم طَرَقَتني نَكبَةٌ بَعدَ نَكبَةٍ   فَفَرَّجتُها عَنّي وَما مَسَّني ضُرُّ \n\n وَلَولا سِناني وَالحُسامُ وَهِمَّتي  لَما ذُكِرَت عَبسٌ وَلا نالَها فَخرُ """
	    
        ,
	
	"""رُبَّ رامٍ مِن بَني ثُعَلٍ  مُتلِجٍ كَفَّيهِ في قُتَرِه \n\nعارِضٍ زَوراءَ مِن نَشمٍ  غَيرُ باناةٍ عَلى وَتَرِه \n\n قَد أَتَتهُ الوَحشُ وارِدَةً  فَتَنَحّى النَزعُ في يَسَرِه"""
        ,"""نَفِّسوا كَربي وَداوُوا عِلَلي  وَاِبرِزوا لي كُلَّ لَيثٍ بَطَلِ \n\n وَاِنهَلوا مِن حَدِّ سَيفي جُرَعاً  مُرَّةً مِثلَ نَقيعِ الحَنظَلِ \n\n وَإِذا المَوتُ بَدا في جَحفَلٍ  فَدَعوني لِلِقاءِ الجَحفَلِ"""
        ,"""بَكَرَت تَعذُلُني وَسطَ الحِلالِ  سَفَهاً بِنتُ ثُوَيرِ بنِ هِلالِ \n\n بَكَرَت تَعذُلُني في أَن رَأَت  إِبِلي نَهباً لِشَربٍ وَفِضالِ \n\n لا تَلوميني فَإِنّي مُتلِفٌ  كُلَّ ما تَحوي يَميني وَشِمالي"""
        ,"""ذادَ عَنى النَومَ هَمٌّ بَعدَ هَمّ  وَمِن الهَمِّ عَناءٌ وَسَقَم \n\n طَرَقَت طَلحَةُ رَحلي بَعدَما  نامَ أَصحابى وَلَيلي لَم أَنَم \n\n طَرَقَتنا ثُمَّ قُلنا إِذ أَتَت  مَرحَباً بِالزَورِ لَمّا أَن أَلَمّ""" 
        ,"""تَقولُ أَلا قَد دَنا نازِحٌ  فِداءٌ لَهُ الطارِفُ التالِدُ \n\n أَخٌ لَم تَكُن أُمُّنا أَمَّه  وَلَكِن أَبونا أَبٌ واحِدُ \n\n تَدارَكَني رَأفَةً حاتِمٌ  فَنِعمَ المُرَبِّبُ وَالوالِدُ"""

        ,

        """أَحارِ بنُ عَمروٍ كَأَنّي خَمِر  وَيَعدو عَلى المَرءِ ما يَأتَمِر \n\n لا وَأَبيكَ اِبنَةَ العامِرِيِّ  لا يَدَّعي القَومُ أَنّي أَفِر \n\n تَميمُ بنُ مُرٍّ وَأَشياعُها  وَكِندَةُ حَولي جَميعاً صُبُر """
	,""" لَعَمرُكَ ما إِن لَهُ صَخرَةً  لَعَمرُكَ ما إِن لَهُ وَزَر """
	,""" أأَلَم تُكسَفِ الشَمسُ وَالبَدرُ وَالـ  ـكَواكِبُ لِلجَبَلِ الواجِبِ  \n\n لِفَقدِ فَضالَةَ لا تَستَوي الـ  ـفُقودُ وَلا خَلَّةُ الذاهِبِ  \n\n أَلَهفاً عَلى حُسنِ أَخلاقِهِ  عَلى الجابِرِ العَظمِ وَالحارِبِ """
	,""" أَجِدّوا النِعالَ لِأَقدامِكُم أَجِدّوا فَوَيهاً لَكُم جَروَلُ  \n\n وَأَبلِغ سَلامانَ إِن جِئتَها  فَلا يَكُ شِبهاً لَها المِغزَلُ \n\n يُكَسّي الأَنامَ وَيُعري أَستَهُ  وَيَنسَلُّ مِن خَلفِهِ الأَسفَلُ """
	,"""تُخَبِّرُني بِالنَجاةِ القَطاةُ  وَقَولُ الغُرابِ لَها شاهِدُ  \n\n تَقولُ أَلا قَد دَنا نازِحٌ  فِداءٌ لَهُ الطارِفُ التالِدُ  \n\n أَخٌ لَم تَكُن أُمُّنا أَمَّه  وَلَكِن أَبونا أَبٌ واحِدُ """


        ,

        """لِمَنِ الدِيارُ غَشِيتُها بِسُحامِ   فَعَمايَتَينِ فَهُضبُ ذي أَقدامِ \n\n فَصَفا الأَطيطِ فَصاحَتَينِ فَغاضِرٍ   تَمشي النِعاجُ بِها مَعَ الآرامِ   دارٌ لِهِندٍ وَالرَبابِ وَفَرتَنى \n\n وَلَميسَ قَبلَ حَوادِثِ الأَيّامِ  عوجا عَلى الطَلَلِ المَحيلِ لِأَنَنا"""
	,"""هَل غادَرَ الشُعَراءُ مِن مُتَرَدَّمِ   أَم هَل عَرَفتَ الدارَ بَعدَ تَوَهُّمِ \n\n يا دارَ عَبلَةَ بِالجَواءِ تَكَلَّمي   وَعَمي صَباحاً دارَ عَبلَةَ وَاِسلَمي \n\n فَوَقَفتُ فيها ناقَتي وَكَأَنَّها  فَدَنٌ لِأَقضِيَ حاجَةَ المُتَلَوِّمِ"""
        ,"""أَبَني زَبيبَةَ ما لِمُهرِكُمُ   مُتَخَدِّداً وَبُطونُكُم عُجرُ \n\n أَلَكُم بِآلاءِ الوَشيجِ إِذا   مَرَّ الشِياهُ بِوَقعِهِ خُبرُ \n\n إِذ لا تَزالُ لَكُم مُغَرغَرَةٌ  تَغلي وَأَعلى لَونِها صَهرُ \n\n لَمّا غَدَوا وَغَدَت سَطيحَتُهُم  مَلأى وَبَطنُ جَوادِهِم صِفرُ"""
	,"""قِف بِالمَنازِلِ إِن شَجَتكَ رُبوعُها   فَلَعَلَّ عَينَكَ تَستَهِلُّ دُموعُها \n\n وَاِسأَل عَنِ الأَظعانِ أَينَ سَرَت بِها   آباؤُها وَمَتى يَكونُ رُجوعُها \n\n دارٌ لِعَبلَةَ شَطَّ عَنكَ مَزارُها   وَنَأَت فَفارَقَ مُقلَتَيكَ هُجوعُها \n\n فَسَقَتكِ يا أَرضَ الشَرَبَّةِ مُزنَةٌ   مُنهَلَّةٌ يَروي ثَراكِ هُموعُها"""
        
	,

        """كَم يُبعِدُ الدَهرُ مَن أَرجو أُقارِبُهُ  عَنّي وَيَبعَثُ شَيطاناً أُحارِبُهُ \n\n فَيا لَهُ مِن زَمانٍ كُلَّما اِنصَرَفَت  صُروفُهُ فَتَكَت فينا عَواقِبُهُ"""
	,"""لا يَحمِلُ الحِقدَ مَن تَعلو بِهِ الرُتَبُ  وَلا يَنالُ العُلا مَن طَبعُهُ الغَضَبُ \n\n وَمَن يِكُن عَبدَ قَومٍ لا يُخالِفُهُم  إِذا جَفوهُ وَيَستَرضي إِذا عَتَبوا"""
        ,"""لَمّا جَفاني أَخِلّائي وَأَسلَمَني   دَهري وَلحمُ عِظامي اليَومَ يُعتَرَقُ \n\n أَقبَلتُ نَحوَ أَبي قابوسَ أَمدَحُهُ   إِنَّ الثَناءَ لَهُ وَالحَمدُ يَتَّفِقُ"""
	,"""نفع قليلٌ إذا نادى الصدى أُصلا   وحانَ منه لبرد الماء تَغريد \n\n وودعوني فقالوا ساعة انطلقوا   أودى فأودى النَدى والحزم والجود"""
        ,"""قد أصبح الحبل من أسماء مصروما   بعد ائتلافٍ وحب كان مكتوما \n\n واستبدلت خلة مني وقد علمت   أن لن أبيت بوادي الخسف مذموما"""

	    
    ],
    "Sea": [
        "بحر الطويل","بحر الطويل" ,"بحر الطويل", "بحر الطويل", "بحر الطويل",
        "بحر الرمل","بحر الرمل","بحر الرمل","بحر الرمل", "بحر الرمل",
        "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب",
        "بحر الكامل", "بحر الكامل", "بحر الكامل", "بحر الكامل",
        "بحر البسيط","بحر البسيط","بحر البسيط","بحر البسيط","بحر البسيط"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)


# Define Functions
def load_data():
    return df

def create_documents(df):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000000)
    marked_text = []
    for i in range(len(df)):
        poem = df['Poem'].iloc[i]
        sea = df['Sea'].iloc[i]
        markdown_text = f'#{sea} : {poem}'
        marked_text.append(markdown.markdown(markdown_text))
    return splitter.create_documents(marked_text)

def create_embedding(documents):
    embeddings = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v0")
    return FAISS.from_documents(documents, embeddings)

def get_credentials():
    return {
        "url": "https://eu-de.ml.cloud.ibm.com",
        "apikey": api_key,  
        "project_id": "11af8977-9294-4e73-a863-b7e37a214840",
    }



prompt_1 = """<s>[INST] <<SYS>>
أنت شاعر عربي فصيح، عليك كتابة قصيدة شعرية باللغة العربية فقط، مع الالتزام بقوانين التشكيل الشعرية التالية وفق بحر المعطى في السؤال وتفعيلاته، مع الحرص على أن تحمل معنى

هذه القواعد يجب عليك الإلتزام بها:

1. الحروف في اللغة العربية:
* الحرف الساكن: يمثل عدم وجود حركات، ويُشير إليه بعلامة السكون ( ْ ).
* الحرف المتحرك: يدل على اتجاه الصوت عند النطق، ويكون إما ضمة ( ُ ) أو فتحة ( َ ) أو كسرة ( ِ ).
* الشدة ( ّ ): تعبر عن تكرار الحرف، حيث يكون الأول ساكنًا والثاني متحركًا.

2. المقاطع العروضية:
* السبب الخفيف: يجمع حرفًا متحركًا ثم حرفًا ساكنًا مثل: لَمْ، عَنْ، كَمْ.
* السبب الثقيل: ويكون عندما يجتمع حرفين متحركين مع بعضهما البعض مثل لَكَ - بِكَ - مَعَ.
* الوتد المجموع: يتكون من حرفين متحركين وآخر ساكن مثل: إِلَى، عَلَى.
* الوتد المفروق: يتكون من حرف متحرك، ثم ساكن، ثم متحرك مثل: أَيْنَ، قَاْمَ.
* الفاصلة الصغرى: وهي تتألف من أربعة أحرف ، الثلاثة الأولى منها متحركة ، والرابع ساكن مثل لَعِبَتْ - فَرِجَتْ - رَجَعَاْ إلى آخره.
* الفاصلة الكبرى: وهي تتألف من خمسة أحرف الأربعة الأولى متحركة والخامس ساكن مثل شَجَرَةٌ - ثَمَرَةٌ ( التنوين عبارة عن حركة يليها ساكن "شَجَرَتُنْ").

3. التفاعيل:
* فَعُوْلُن: وهو يتألف من فعو + لن (وتد مجموع + سبب خفيف)
* فَاْعِلُنْ: وهو يتألف من فا + علن (سبب خفيف + وتد مجموع)
* مَفَاْعِيْلُنْ: وهو يتألف من مفا + عي + لن (وتد مجموع + سبب خفيف + سبب خفيف)
* مُفَاْعَلَتُنْ: وهو يتألف من مفا + علتن (وتد مجموع + فاصلة صغرى)
* مُتَفَاْعِلُنْ: وهو يتألف من متفا + علن (فاصلة صغرى + وتد مجموع)
* مَفْعُوْلَاْتُ: وهو يتألف من مف + عو + لات (سبب خفيف + سبب خفيف + وتد مفروق)
* مُسْتَفْعِلُنْ: وهو يتألف من مس + تف + علن (سبب خفيف + سبب خفيف + وتد مجموع) أو مس + تفع + لن (سبب خفيف + وتد مفروق + سبب خفيف)
* فَاْعِلَاْتُنْ: وهو يتألف من فا + علا + تن (سبب خفيف + وتد مجموع + سبب خفيف) أو فاع + لا + تن (وتد مفروق + سبب خفيف + سبب خفيف)

4. البحور:
* بحر الطويل: يجب أن يتبع أي شطر في قصائد بحر الطويل
سياق التفعيلة: فَعُوْلُنْ مَفَاْعِيْلُنْ فَعُوْلُنْ مَفَاْعِيلُنْ

* بحر الرمل: يجب أن يتبع أي شطر في قصائد بحر الرمل
سياق التفعيلة: فَاْعِلاتُنْ فَاْعِلاتُنْ فَاْعِلاتُنْ

* بحر المتقارب: يجب أن يتبع أي شطر في قصائد بحر المتقارب
سياق التفعيلة: فَعُوْلُنْ فَعُوْلُنْ فَعُوْلُنْ فَعُوْلُ

* بحر الكامل: يجب أن يتبع أي شطر في قصائد بحر الكامل
سياق التفعيلة: مُتَفَاْعِلُنْ مُتَفَاْعِلُنْ مُتَفَاْعِلُنْ

* بحر البسيط: يجب أن يتبع أي شطر في قصائد بحر البسيط
سياق التفعيلة: مُسْتَفْعِلُنْ فَاْعِلُنْ مُسْتَفْعِلُنْ فَاْعِلُنْ
"""

def generate_poetry_response(query, threshold, model):
    results = arabic_VDB.similarity_search_with_score(query, k=2) # you can add k this is the number of the rag context
    context_text = "\n\n".join([doc.page_content for doc, score in results if score > threshold])
    input_with_rag = """{0}

لتوسيع مدارك فهمك يمكنك الإستلهام من هذه الأمثلة:
{1}

انشأ القصيدة بناءً على هذا الطلب: {2} [/INST]""".format(prompt_1, context_text, query)
    response = model.generate(input_with_rag)['results'][0].get('generated_text')
    return response, context_text


# Streamlit App Start
st.title("أهلا بكم في ضيافة الشاعر النابغة المِلساني")

st.write("هنا تستطيع سؤال الشاعر ملسان عن أبيات او إنشاء قصائد من بحور متعددة من اختياركم")

# get API key
api_key = st.text_input("أدخل مفتاح الاستخدام")  

options = ["انشاء قصيدة", "اكمال قصيدة", "شرح قصيدة"]
selected_fruit = st.selectbox("اخر من خدمات ملسان", options)


if selected_fruit == "انشاء قصيدة":
    prompt_1 = prompt_1 + """ 
	عند انشاء القصيدة يجب عليك كتابتها بناءً على هذه الشروط:
	1) الإلتزام بالقواعد المذكورة اعلاه
	2) انشاء قصيدتك الخاصة
	3) القصيدة مُستلهمة بطريقة المُتنبي وليست بكلماته!
	4) عدم ذكر التفعيلات بعد البيت ويجب استخدامها في بناء وزن البيت الشعري
	5) لا يجب عليك استخدام الأمثلة المذكورة ووضعها في القصيدة انما يتم الإستلهام منها فقط!
	<</SYS>>"""
elif selected_fruit == "اكمال قصيدة":
	prompt_1 = prompt_1 + """ 
	عند اكمال القصيدة يجب عليك اكمالها بناءً على هذه الشروط:
	1) الإلتزام بالقواعد المذكورة اعلاه
	2) اكمال المعنى في القصيدة
	3) عدم ذكر التفعيلات بعد البيت ويجب استخدامها في بناء وزن البيت الشعري
	4) لا يجب عليك استخدام الأمثلة المذكورة ووضعها في القصيدة انما يتم الإستلهام منها فقط!
	<</SYS>>"""
    
else: prompt_1 = prompt_1 + """ 
	عند شرح القصيدة يجب عليك شرحها بناءً على هذه الشروط:
	1) الإلتزام بالقواعد المذكورة اعلاه
	2) توضيح المعنى في القصيدة
	3) ذكر التفعيلات الخاصة بالقصيدة ونوعها ومن اي بحر هي
	<</SYS>>"""
#status.markdown(f'<div class="custom-text">{prompt_1}</div>', unsafe_allow_html=True)
# User Input
query = st.text_input("أكتب طلبك لإنشاء قصيدة من أحد البحور الشعرية ")
threshold = st.slider("أختر نسبة التقارب المطلوبة:", 0.0, 1.0, 0.9)


# Process Data and Display Results
if st.button("أطلق العنان"):
    st.write("Generated Poetry:")
    status = st.empty()
	
    status.markdown('<div class="custom-text">يتم الإبداع...</div>', unsafe_allow_html=True)
    documents = create_documents(df) 
    arabic_VDB = create_embedding(documents)
	
    model_id = "sdaia/allam-1-13b-instruct"
    parameters = { 
	"decoding_method": "greedy", 
	"max_new_tokens": 800, 
	"repetition_penalty": 1 
	}
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=get_credentials(),
	project_id ="11af8977-9294-4e73-a863-b7e37a214840",
    )
    response , rag = generate_poetry_response(query, threshold, model)
    status.write(f'<div class="custom-text">{response}</div>', unsafe_allow_html=True)
