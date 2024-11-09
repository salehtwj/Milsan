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
        font-size: 24px;
    }

    /* Explaination container */
    li > ul {
        background-color: 'rgba(255, 255, 255, 0.2)';
    }

    /* Explaination poem container */
    li > p {
        background-color: 'yellow';
        padding: '5px';
        color: 'black';
        width: 'fit-content
    }

    /* Explaination bulletpoints text */
    li > ul > li {
        font-size: '18px';
        font-weight: 'bold;
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

# Retrieval-Augmented Generation (RAG)

# Load Data
DATA = {
    "Poem": [
        """قِفا نَبكِ مِن ذِكرى حَبيبٍ وَعِرفانِ  وَرَسمٍ عَفَت آياتُهُ مُنذُ أَزمانِ \n\n أَتَت حُجَجٌ بَعدي عَلَيها فَأَصبَحَت  كَخَطِّ زَبورٍ في مَصاحِفِ رُهبانِ""",
        """ أَعِنّي عَلى بَرقٍ أَراهُ وَميضِ  يُضيءُ حَبِيّاً في شَماريخَ بيضِ  \n\n وَيَهدَأُ تاراتٍ سَناهُ وَتارَةً  يَنوءُ كَتَعتابِ الكَسيرِ المَهيضِ  \n\n  وَتَخرُجُ مِنهُ لامِعاتٌ كَأَنَّها أَكُفٌّ تَلَقّى الفَوزَ عِندَ المَفيضِ """,
        """ طَرِبتَ وَهاجَتكَ الظِباءُ السَوارِحُ غَداةَ غَدَت مِنها سَنيحٌ وَبارِحُ  \n\n  تَغالَت بِيَ الأَشواقُ حَتّى كَأَنَّما  بِزَندَينِ في جَوفي مِنَ الوَجدِ قادِحُ  \n\n وَقَد كُنتَ تُخفي حُبَّ سَمراءَ حِقبَةَ فَبُح لِانَ مِنها بِالَّذي أَنتَ بائِحُ """,
        """ أُعاتِبُ دَهراً لا يَلينُ لِعاتِبِ وَأَطلُبُ أَمناً مِن صُروفِ النَوائِبِ \n\n وَتوعِدُني الأَيّامُ وَعداً تَغُرُّني وَأَعلَمُ حَقّاً أَنَّهُ وَعدُ كاذِبِ  \n\n خَدَمتُ أُناساً وَاِتَّخَذتُ أَقارِباً لِعَوني وَلَكِن أَصبَحوا كَالعَقارِبِ""",
        """ دَهَتني صُروفُ الدَهرِ وَاِنتَشَبَ الغَدرُ   وَمَن ذا الَّذي في الناسِ يَصفو لَهُ الدَهرُ  \n\n  وَكَم طَرَقَتني نَكبَةٌ بَعدَ نَكبَةٍ   فَفَرَّجتُها عَنّي وَما مَسَّني ضُرُّ \n\n وَلَولا سِناني وَالحُسامُ وَهِمَّتي  لَما ذُكِرَت عَبسٌ وَلا نالَها فَخرُ """,

        """أَحارِ بنُ عَمروٍ كَأَنّي خَمِر  وَيَعدو عَلى المَرءِ ما يَأتَمِر \n\n لا وَأَبيكَ اِبنَةَ العامِرِيِّ  لا يَدَّعي القَومُ أَنّي أَفِر \n\n تَميمُ بنُ مُرٍّ وَأَشياعُها  وَكِندَةُ حَولي جَميعاً صُبُر """,
        """ لَعَمرُكَ ما إِن لَهُ صَخرَةً  لَعَمرُكَ ما إِن لَهُ وَزَر """,
        """ أأَلَم تُكسَفِ الشَمسُ وَالبَدرُ وَالـ  ـكَواكِبُ لِلجَبَلِ الواجِبِ  \n\n لِفَقدِ فَضالَةَ لا تَستَوي الـ  ـفُقودُ وَلا خَلَّةُ الذاهِبِ  \n\n أَلَهفاً عَلى حُسنِ أَخلاقِهِ  عَلى الجابِرِ العَظمِ وَالحارِبِ """,
        """ أَجِدّوا النِعالَ لِأَقدامِكُم أَجِدّوا فَوَيهاً لَكُم جَروَلُ  \n\n وَأَبلِغ سَلامانَ إِن جِئتَها  فَلا يَكُ شِبهاً لَها المِغزَلُ \n\n يُكَسّي الأَنامَ وَيُعري أَستَهُ  وَيَنسَلُّ مِن خَلفِهِ الأَسفَلُ """,
        """تُخَبِّرُني بِالنَجاةِ القَطاةُ  وَقَولُ الغُرابِ لَها شاهِدُ  \n\n تَقولُ أَلا قَد دَنا نازِحٌ  فِداءٌ لَهُ الطارِفُ التالِدُ  \n\n أَخٌ لَم تَكُن أُمُّنا أَمَّه  وَلَكِن أَبونا أَبٌ واحِدُ """,

        """كَم يُبعِدُ الدَهرُ مَن أَرجو أُقارِبُهُ  عَنّي وَيَبعَثُ شَيطاناً أُحارِبُهُ \n\n فَيا لَهُ مِن زَمانٍ كُلَّما اِنصَرَفَت  صُروفُهُ فَتَكَت فينا عَواقِبُهُ""",
        """لا يَحمِلُ الحِقدَ مَن تَعلو بِهِ الرُتَبُ  وَلا يَنالُ العُلا مَن طَبعُهُ الغَضَبُ \n\n وَمَن يِكُن عَبدَ قَومٍ لا يُخالِفُهُم  إِذا جَفوهُ وَيَستَرضي إِذا عَتَبوا""",
        """لَمّا جَفاني أَخِلّائي وَأَسلَمَني   دَهري وَلحمُ عِظامي اليَومَ يُعتَرَقُ \n\n أَقبَلتُ نَحوَ أَبي قابوسَ أَمدَحُهُ   إِنَّ الثَناءَ لَهُ وَالحَمدُ يَتَّفِقُ""",
        """نفع قليلٌ إذا نادى الصدى أُصلا   وحانَ منه لبرد الماء تَغريد \n\n وودعوني فقالوا ساعة انطلقوا   أودى فأودى النَدى والحزم والجود""",
        """قد أصبح الحبل من أسماء مصروما   بعد ائتلافٍ وحب كان مكتوما \n\n واستبدلت خلة مني وقد علمت   أن لن أبيت بوادي الخسف مذموما""",

        """هاجَ رَسمٌ دارِسٌ طَرَباً   فطويلا ظَللّتَ مُكتَئِبا \n\n أن رأَيتَ الدارَ موحِشَةً   بِلغاطٍ كَم لَها رَجَبا \n\n دارَ هِندٍ بالسِتارِ وَقَد   رَثَّ حَبلُ العهد فاِنقَضَبا""",
        """نَّ بِالشّعبِ الذي دونَ سلعٍ   لَقتيلاً دَمُه ما يُطَلُّ \n\n خَلَّفَ العِبءَ علَيَّ وَوَلّى   أَنَا بِالعِبءِ له مُستَقِلُّ \n\n وَوَراءَ الثَّأرِ منّي ابنُ أُختٍ   مَصِعٌ عُقدَتُهُ ما تُحَلُّ""",
        """مُطرِقٌ يَرشُح مَوتاً كَما   أطرَقَ أَفعى يَنفُثُ السمَّ صِلُّ \n\n خَبَرٌ ما نابَنا مصمَئِلٌّ   جَلَّ حَتّى دَقَّ فِيه الأَجَلُّ \n\n بَزَّنِي الدَّهرُ وكانَ غَشُوماً   بِأبِيٍّ جَارُهُ ما يُذَلُّ""",
        """إِنَّ بِالشّعبِ الذي دونَ سلعٍ   لَقتيلاً دَمُه ما يُطَلُّ \n\n\ خَلَّفَ العِبءَ علَيَّ وَوَلّى   أَنَا بِالعِبءِ له مُستَقِلُّ \n\n وَوَراءَ الثَّأرِ منّي ابنُ أُختٍ   مَصِعٌ عُقدَتُهُ ما تُحَلُّ""",

        """تَاللَهِ لا يَذهَبُ شَيخي باطِلا   حَتّى أُبيرَ مالِكاً وَكاهِلا \n\n القاتِلينَ المَلِكَ الحُلاحِلا   خَيرَ مَعَدٍّ حَسَباً وَنائِلا \n\n يا لَهفَ هِندٍ إِذ خَطِئنَ كاهِلا   نَحنُ جَلَبنا القُرَّحَ القَوافِلا""",
        """قُلتُ مَنِ القَومُ فَقالوا سَفَرَه   وَالقَومُ كَعبٌ يَبتَغونَ المُنكَرَه \n\n قُلتُ لِكَعبٍ وَالقَنا مُشتَجِرَه   تَعَلَّمي يا كَعبُ وَاِمشي مُبصِرَه""",
        """الحمد لله على السراء   حمد شكور خالص الثناء \n\n حمداً على الأحسان والأفضال   بلغنا نهاية الامال \n\n نلنا المنى في أرض سامراء   حيث الندى ومعدن الالاء""",
        """ذاكَ الثُوَيرُ فَما أُحِبُّ بِفَضلِهِ  عِندَ التَفاضُلِ فَضلَ قَومٍ أَفضَلا \n\n ما بِامرِئٍ مِن ضُؤلَةٍ في وائِلٍ   وَرِثَ الثُوَيرَ وَمالِكاً وَمُهَلهِلا \n\n خالي بِذي بَقَرٍ حَمى أَصحابَهُ   وَشَرى بِحُسنِ حَديثِهِ أَن يُقتَلا""",

        """قولا لِدودانَ عَبيدِ العَصا  ما غَرَّكُم بِالأَسَدِ الباسِلِ \n\n صُمَّ صَداها وَعَفا رَسمُها  وَاِستَعجَمَت عَن مَنطِقِ السائِلِ \n\n يا دارَ ماوِيَّةَ بِالحائِلِ  فَالسُهبِ فَالخَبتَينِ مِن عاقِلِ""",
        """باتوا يُصيبُ القَومُ ضَيفاً لَهُم  حَتّى إِذا ما لَيلُهُم أَظلَما \n\n إِذ قالَ عَمروٌ لِبَني مالِكٍ  لا تَعجَلوا المِرَّةَ أَن تُحكَما \n\n كانَ بَنو الأَبرَصِ أَقرانَكُم  فَأَدرَكوا الأَحدَثَ وَالأَقدَما""",
        """قُلتُ لِعَمروٍ حينَ أَرسَلتُهُ  وَقَد حَبا مِن دُونِهِ عالِجُ \n\n ولا قَعيدٌ أَغَضَبٌ قَرنُهُ  هاجَ لَهُ مِن مَرتَعٍ هائِجُ \n\n يا أَيُّها المُزمِعُ ثُمَّ اِنثَنى  لا يَثنِكَ الحازي وَلا الشاحِجُ""",
        """يَزخَرُ في أَقطارِهِ مُغدِقٌ  بِجَما فَتيِهِ الشوعَ وَالغِريَفِ \n\n مُعرَورِفٌ أَسبَلَ جُبَّارَهُ  أَسوَدُ كَالغابَةِ مُغدَودِفِ \n\n إِذا جَمادى مَنَعَت قِطرَها  زانَ جَناني عَطنٌ مُغضَفِ""",

        """قلبي قلق جسمي شحب  ما بهما قسراً رقشا \n\n قولوا لهموا ما قصد هموا  وهواه لقلب الصب حشى \n\n الصبر ثوى لما حجبوا  بدراً قد أذاب هواه الحشى""",
        """يا نفسُ أحيَي تَصِلِي أمَلاً  عيشي رَجَباً تَرَي عَجَبا \n\n ذَرتِ السِّتونَ بُرادَتَها  فِي مِسكِ عِذارَكِ فاشتَهَبَا \n\n أَبعَيدَ الشّبابِ هَوى وَصِبا  كلا لا لَهوَ ولا لَعِبا""",
        """وعَنَت لعزائِمكم عربٌ  تشقى بصوارِمَها العجَمُ \n\n وهَمَت ديمٌ من راحَتِكُم  هيهاتَ تُساجِلُها الديَمُ \n\n شملَت ببقائِكِم النعَمُ  وسمتِ برجائِكُمُ الهِمَمُ""",
        """أَبِنَظرَةِ عَينٍ عَن خَطأٍ  عَرَضَت بِالعَمدِ يُراقُ دَمي \n\n فَتعالَي غَيرَ مُدافِعَةٍ  نَقصُص رُؤياكِ عَلى حَكَمِ \n\n مَن ذا أَفتاكِ بِسَفكِ دَمي  يا غُرَّةَ حَيِّ بَني جُشَمِ"""
    ],

    "Sea": [
        "بحر الطويل", "بحر الطويل", "بحر الطويل", "بحر الطويل", "بحر الطويل",

        "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب",

        "بحر البسيط", "بحر البسيط", "بحر البسيط", "بحر البسيط", "بحر البسيط",

        "بحر المديد", "بحر المديد", "بحر المديد", "بحر المديد",

        "بحر الرجز", "بحر الرجز", "بحر الرجز", "بحر الرجز",

        "بحر السريع", "بحر السريع", "بحر السريع", "بحر السريع",

        "بحر المتدارك", "بحر المتدارك", "بحر المتدارك", "بحر المتدارك"    
    ]
}

df = pd.DataFrame(DATA)

def create_documents(df):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000000)
    marked_text = []

    for i in range(len(df)):
        poem = df['Poem'].iloc[i]
        sea = df['Sea'].iloc[i]

        marked_text.append(markdown.markdown(f'#{sea} : {poem}'))

    return splitter.create_documents(marked_text)

def create_embedding(documents):
    embeddings = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v0")
    return FAISS.from_documents(documents, embeddings)

poemConstraints = """1. الحروف في اللغة العربية:
* الحرف الساكن: يمثل عدم وجود حركات، ويُشير إليه بعلامة السكون ( ْ ).
* الحرف المتحرك: يدل على اتجاه الصوت عند النطق، ويكون إما ضمة ( ُ ) أو فتحة ( َ ) أو كسرة ( ِ ).
* الشدة ( ّ ): تعبر عن تكرار الحرف، حيث يكون الأول ساكنًا والثاني متحركًا.

2. التفاعيل: الرجاء عدم استخدام هذه التفاعيل أبدا
* فَعُوْلُن
* فَاْعِلُنْ
* مُسْتَفْعِلُنْ
* فَاْعِلَاْتُنْ

3. البحور: لا تكتبها ابدا
* بحر الطويل: يجب أن يتبع أي شطر في قصائد بحر الطويل سياق التفعيلة: فَعُوْلُنْ مَفَاْعِيْلُنْ فَعُوْلُنْ مَفَاْعِيلُنْ
* بحر المتقارب: يجب أن يتبع أي شطر في قصائد بحر المتقارب سياق التفعيلة: فَعُوْلُنْ فَعُوْلُنْ فَعُوْلُنْ فَعُوْلُ
* بحر البسيط: يجب أن يتبع أي شطر في قصائد بحر البسيط سياق التفعيلة: مُسْتَفْعِلُنْ فَاْعِلُنْ مُسْتَفْعِلُنْ فَاْعِلُنْ
* بحر المديد: يجب أن يتبع أي شطر في قصائد بحر المديد هذه التفعيلة: فَاْعِلَاْتُنْ فَاْعِلُنْ فَاْعِلَاْتُنْ 
* بحر الرجز: يجب أن يتبع أي شطر في قصائد بحر الرجز هذه التفعيلة: مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ  
* بحر السريع: يجب أن يتبع أي شطر في قصائد بحر السريع هذه التفعيلة: مُسْتَفْعِلُنْ مُسْتَفْعِلُنْ فَاْعِلُنْ 
* بحر المتدارك: يجب أن يتبع أي شطر في قصائد بحر المتدارك هذه التفعيلة: فَاعِلُنْ فَاعِلُنْ فَاعِلُنْ فَاعِلُنْ"""


agentTypes = {
    "انشاء قصيدة": "انت شاعر عربي فصيح، عليك كتابة قصيدة شعرية باللغة العربية فقط",
    "اكمال قصيدة": "انت شاعر عربي فصيح، ومهمتك الأساسية هي اكمال القصائد الشعرية باللغة العربية فقط",
}

def constructAgentPrompt(type):
    if(type == "شرح قصيدة"):
        return f"""انت شاعر عربي فصيح، ومهمتك الأساسية هي توضيح القصيدة الشعرية بطريقة لغوية مبسطة

هذه القواعد يجب عليك الإلتزام بها عند توضيح القصيدة:
{poemConstraints}

يجب عليك الإلتزام بالشروط الآتية اثناء شرح القصيدة:
* الالتزام بالقواعد المذكورة أعلاه!!!
* يجب عليك توضيح القصيدة بطريقة مبسطة وواضحة للمستخدم
* لا يجب التفصيل في القواعد الشعرية المعقدة، انما توضيح المعنى من الأبيات الشعرية
* يجب عليك كتابة البيت والمعنى منه في نقاط مبسطة"""
    
    elif(type == "رد على قصيدة"):
        return f"""انت شاعر عربي فصيح، مهمتك مساعدة المستخدم في انشاء رد على قصيدة او بيت شعري
        
هذه القواعد يجب عليك الإلتزام بها اثناء كتابة الرد
{poemConstraints}

يجب عليك الإلتزام بالشروط الآتية عند كتابة الرد:
* الالتزام بالقواعد المذكورة أعلاه!!!
* يجب ان تفهم معاني القصيدة وافكارها قبل بناء الرد
* يجب عليك معرفة نوع القصيدة (إيجابية، سلبية)
* يجب الإلتزام بالوزن والقافية إذا كانت القصيدة موزونة يجب ان يكون الرد كذلك مع مراعاة القافية
* يجب ان يكون الرد بلمستك الخاصة الإبداعية سواءً في الفكرة او الأسلوب
* يجب ان يكون الرد محترمًا ولا توجد به اساءة للشاعر او القصيدة الأصلية
* يجب ان يكون الرد متناسقًا مع موضوع القصيدة"""

    return f"""{agentTypes.get(type)}

مع الالتزام بقوانين التشكيل الشعرية التالية وفق بحر المعطى في السؤال وتفعيلاته، مع الحرص على أن تحمل معنى

هذه القواعد يجب عليك الإلتزام بها دون تكرار أي بيت:
{poemConstraints}

عند انشاء القصيدة يجب عليك كتابتها بناءً على هذه الشروط:
* الالتزام بالقواعد المذكورة أعلاه!!!
* انشاء قصيدتك الخاصة!
* القصيدة مُستلهمة بطريقة المُتنبي أو النابغة الذبياني أو عنترة بن شداد أو امرؤ القيس وليست بكلماته!
* عدم ذكر التفعيلات بعد البيت ويجب استخدامها في بناء وزن البيت الشعري!
* لا يجب عليك استخدام الأمثلة المذكورة ووضعها في القصيدة انما يتم الاستلهام منها فقط!
* لا يجب عليك ذكر اسم البحر المستخدم في القصيدة!
* القصيدة يجب ان لا تقل عن بيتين شعرية!"""

def explainPoetry(query, model):    
    PROMPT = """<s>[INST] <<SYS>>
{0}
<</SYS>>

اشرح القصيدة الآتية: {1} [/INST]""".format(constructAgentPrompt("شرح قصيدة"), query)

    return model.generate(PROMPT)['results'][0].get('generated_text')

def respondToPoetry(query, model):    
    PROMPT = """<s>[INST] <<SYS>>
{0}
<</SYS>>

رد على هذه القصيدة: {1} [/INST]""".format(constructAgentPrompt("رد على قصيدة"), query)

    return model.generate(PROMPT)['results'][0].get('generated_text')


def continuePoetry(query, model):
    PROMPT = """<s>[INST] <<SYS>>
{0}
<</SYS>>

اكمل القصيدة الآتية: {1} [/INST]""".format(constructAgentPrompt("اكمال قصيدة"), query)

    return model.generate(PROMPT)['results'][0].get('generated_text')

def generatePoetry(query, model):
    results = arabic_VDB.similarity_search_with_score(query, k=3)
    related_poems = "\n\n".join([doc.page_content for doc, score in results if score > 0.9])

    PROMPT = """<s>[INST] <<SYS>>
{0}

يمكنك الإستلهام من هذه الأبيات الشعرية (لا يجب عليك استخدام الكلمات المذكورة في انشاء القصيدة!!):
{1}
<</SYS>>

اكتب القصيدة الآتية: {2} [/INST]""".format(constructAgentPrompt("انشاء قصيدة"), related_poems, query)
    return model.generate(PROMPT)['results'][0].get('generated_text')

# Streamlit App Start
st.title("أهلا بكم في ضيافة الشاعر النابغة المِلساني")

st.write("هنا تستطيع سؤال الشاعر ملسان عن أبيات او إنشاء قصائد من بحور متعددة من اختياركم")

# get API key
api_key = st.text_input("أدخل مفتاح الاستخدام")  

# User Input
options = ["انشاء قصيدة", "اكمال قصيدة", "شرح قصيدة", "رد على قصيدة"]
selectedType = st.selectbox("اختر من خدمات ملسان", options)

if(selectedType):
    labels = {
        "انشاء قصيدة": "اكتب طلبك لإنشاء القصيدة",
        "رد على قصيدة": "القصيدة المُراد الرد عليها",
        "شرح قصيدة": "اكتب القصيدة التي تريد شرحها",
        "اكمال قصيدة": "اكتب القصيدة التي تريد إكمالها"
    }

    query = (st.text_input if selectedType == "انشاء قصيدة" else st.text_area)(labels.get(selectedType))

    # Process Data and Display Results
    if st.button("أطلق العنان"):
        status = st.empty()
        
        status.markdown('<div class="custom-text">يتم الإبداع...</div>', unsafe_allow_html=True)

        documents = create_documents(df) 
        arabic_VDB = create_embedding(documents)

        model = Model(
            model_id = "sdaia/allam-1-13b-instruct",
            params = { 
                "decoding_method": "sample",
                "max_new_tokens": 200 if selectedType != "شرح قصيدة" else 1000,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 1,
                "repetition_penalty": 1
            },

            credentials = {
                "url": "https://eu-de.ml.cloud.ibm.com",
                "apikey": api_key,  
                "project_id": "11af8977-9294-4e73-a863-b7e37a214840",
            },

            project_id = "11af8977-9294-4e73-a863-b7e37a214840",
        )

        response = "يوجد خطأ!"

        if(selectedType == "انشاء قصيدة"):
            response = generatePoetry(query, model)
        elif (selectedType == "اكمال قصيدة"):
            response = continuePoetry(query, model)
        elif (selectedType == "شرح قصيدة"):
            response = explainPoetry(query, model)
        elif (selectedType == "رد على قصيدة"):
            response = respondToPoetry(query, model)

        status.write(f'<div class="custom-text">{response}</div>', unsafe_allow_html=True)
