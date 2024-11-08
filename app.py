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
        """قِفا نَبكِ مِن ذِكرى حَبيبٍ وَعِرفانِ  وَرَسمٍ عَفَت آياتُهُ مُنذُ أَزمانِ \n\n أَتَت حُجَجٌ بَعدي عَلَيها فَأَصبَحَت  كَخَطِّ زَبورٍ في مَصاحِفِ رُهبانِ""",
        """ أَعِنّي عَلى بَرقٍ أَراهُ وَميضِ  يُضيءُ حَبِيّاً في شَماريخَ بيضِ  \n\n وَيَهدَأُ تاراتٍ سَناهُ وَتارَةً  يَنوءُ كَتَعتابِ الكَسيرِ المَهيضِ  \n\n  وَتَخرُجُ مِنهُ لامِعاتٌ كَأَنَّها أَكُفٌّ تَلَقّى الفَوزَ عِندَ المَفيضِ """,
        """ طَرِبتَ وَهاجَتكَ الظِباءُ السَوارِحُ غَداةَ غَدَت مِنها سَنيحٌ وَبارِحُ  \n\n  تَغالَت بِيَ الأَشواقُ حَتّى كَأَنَّما  بِزَندَينِ في جَوفي مِنَ الوَجدِ قادِحُ  \n\n وَقَد كُنتَ تُخفي حُبَّ سَمراءَ حِقبَةَ فَبُح لِانَ مِنها بِالَّذي أَنتَ بائِحُ """,
        """ أُعاتِبُ دَهراً لا يَلينُ لِعاتِبِ وَأَطلُبُ أَمناً مِن صُروفِ النَوائِبِ \n\n وَتوعِدُني الأَيّامُ وَعداً تَغُرُّني وَأَعلَمُ حَقّاً أَنَّهُ وَعدُ كاذِبِ  \n\n خَدَمتُ أُناساً وَاِتَّخَذتُ أَقارِباً لِعَوني وَلَكِن أَصبَحوا كَالعَقارِبِ""",
        """ دَهَتني صُروفُ الدَهرِ وَاِنتَشَبَ الغَدرُ   وَمَن ذا الَّذي في الناسِ يَصفو لَهُ الدَهرُ  \n\n  وَكَم طَرَقَتني نَكبَةٌ بَعدَ نَكبَةٍ   فَفَرَّجتُها عَنّي وَما مَسَّني ضُرُّ \n\n وَلَولا سِناني وَالحُسامُ وَهِمَّتي  لَما ذُكِرَت عَبسٌ وَلا نالَها فَخرُ """,
        """رُبَّ رامٍ مِن بَني ثُعَلٍ  مُتلِجٍ كَفَّيهِ في قُتَرِه \n\nعارِضٍ زَوراءَ مِن نَشمٍ  غَيرُ باناةٍ عَلى وَتَرِه \n\n قَد أَتَتهُ الوَحشُ وارِدَةً  فَتَنَحّى النَزعُ في يَسَرِه""",
        """نَفِّسوا كَربي وَداوُوا عِلَلي  وَاِبرِزوا لي كُلَّ لَيثٍ بَطَلِ \n\n وَاِنهَلوا مِن حَدِّ سَيفي جُرَعاً  مُرَّةً مِثلَ نَقيعِ الحَنظَلِ \n\n وَإِذا المَوتُ بَدا في جَحفَلٍ  فَدَعوني لِلِقاءِ الجَحفَلِ""",
        """بَكَرَت تَعذُلُني وَسطَ الحِلالِ  سَفَهاً بِنتُ ثُوَيرِ بنِ هِلالِ \n\n بَكَرَت تَعذُلُني في أَن رَأَت  إِبِلي نَهباً لِشَربٍ وَفِضالِ \n\n لا تَلوميني فَإِنّي مُتلِفٌ  كُلَّ ما تَحوي يَميني وَشِمالي""",
        """ذادَ عَنى النَومَ هَمٌّ بَعدَ هَمّ  وَمِن الهَمِّ عَناءٌ وَسَقَم \n\n طَرَقَت طَلحَةُ رَحلي بَعدَما  نامَ أَصحابى وَلَيلي لَم أَنَم \n\n طَرَقَتنا ثُمَّ قُلنا إِذ أَتَت  مَرحَباً بِالزَورِ لَمّا أَن أَلَمّ""",
        """تَقولُ أَلا قَد دَنا نازِحٌ  فِداءٌ لَهُ الطارِفُ التالِدُ \n\n أَخٌ لَم تَكُن أُمُّنا أَمَّه  وَلَكِن أَبونا أَبٌ واحِدُ \n\n تَدارَكَني رَأفَةً حاتِمٌ  فَنِعمَ المُرَبِّبُ وَالوالِدُ""",
        """أَحارِ بنُ عَمروٍ كَأَنّي خَمِر  وَيَعدو عَلى المَرءِ ما يَأتَمِر \n\n لا وَأَبيكَ اِبنَةَ العامِرِيِّ  لا يَدَّعي القَومُ أَنّي أَفِر \n\n تَميمُ بنُ مُرٍّ وَأَشياعُها  وَكِندَةُ حَولي جَميعاً صُبُر """,
        """ لَعَمرُكَ ما إِن لَهُ صَخرَةً  لَعَمرُكَ ما إِن لَهُ وَزَر """,
        """ أأَلَم تُكسَفِ الشَمسُ وَالبَدرُ وَالـ  ـكَواكِبُ لِلجَبَلِ الواجِبِ  \n\n لِفَقدِ فَضالَةَ لا تَستَوي الـ  ـفُقودُ وَلا خَلَّةُ الذاهِبِ  \n\n أَلَهفاً عَلى حُسنِ أَخلاقِهِ  عَلى الجابِرِ العَظمِ وَالحارِبِ """,
        """ أَجِدّوا النِعالَ لِأَقدامِكُم أَجِدّوا فَوَيهاً لَكُم جَروَلُ  \n\n وَأَبلِغ سَلامانَ إِن جِئتَها  فَلا يَكُ شِبهاً لَها المِغزَلُ \n\n يُكَسّي الأَنامَ وَيُعري أَستَهُ  وَيَنسَلُّ مِن خَلفِهِ الأَسفَلُ """,
        """تُخَبِّرُني بِالنَجاةِ القَطاةُ  وَقَولُ الغُرابِ لَها شاهِدُ  \n\n تَقولُ أَلا قَد دَنا نازِحٌ  فِداءٌ لَهُ الطارِفُ التالِدُ  \n\n أَخٌ لَم تَكُن أُمُّنا أَمَّه  وَلَكِن أَبونا أَبٌ واحِدُ """,
        """لِمَنِ الدِيارُ غَشِيتُها بِسُحامِ   فَعَمايَتَينِ فَهُضبُ ذي أَقدامِ \n\n فَصَفا الأَطيطِ فَصاحَتَينِ فَغاضِرٍ   تَمشي النِعاجُ بِها مَعَ الآرامِ   دارٌ لِهِندٍ وَالرَبابِ وَفَرتَنى \n\n وَلَميسَ قَبلَ حَوادِثِ الأَيّامِ  عوجا عَلى الطَلَلِ المَحيلِ لِأَنَنا""",
        """هَل غادَرَ الشُعَراءُ مِن مُتَرَدَّمِ   أَم هَل عَرَفتَ الدارَ بَعدَ تَوَهُّمِ \n\n يا دارَ عَبلَةَ بِالجَواءِ تَكَلَّمي   وَعَمي صَباحاً دارَ عَبلَةَ وَاِسلَمي \n\n فَوَقَفتُ فيها ناقَتي وَكَأَنَّها  فَدَنٌ لِأَقضِيَ حاجَةَ المُتَلَوِّمِ""",
        """أَبَني زَبيبَةَ ما لِمُهرِكُمُ   مُتَخَدِّداً وَبُطونُكُم عُجرُ \n\n أَلَكُم بِآلاءِ الوَشيجِ إِذا   مَرَّ الشِياهُ بِوَقعِهِ خُبرُ \n\n إِذ لا تَزالُ لَكُم مُغَرغَرَةٌ  تَغلي وَأَعلى لَونِها صَهرُ \n\n لَمّا غَدَوا وَغَدَت سَطيحَتُهُم  مَلأى وَبَطنُ جَوادِهِم صِفرُ""",
        """قِف بِالمَنازِلِ إِن شَجَتكَ رُبوعُها   فَلَعَلَّ عَينَكَ تَستَهِلُّ دُموعُها \n\n وَاِسأَل عَنِ الأَظعانِ أَينَ سَرَت بِها   آباؤُها وَمَتى يَكونُ رُجوعُها \n\n دارٌ لِعَبلَةَ شَطَّ عَنكَ مَزارُها   وَنَأَت فَفارَقَ مُقلَتَيكَ هُجوعُها \n\n فَسَقَتكِ يا أَرضَ الشَرَبَّةِ مُزنَةٌ   مُنهَلَّةٌ يَروي ثَراكِ هُموعُها""",
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
        """رُبَّما ضَربَةٍ بسيفٍ صَقِيلٍ   دُونَ بُصرَى وَطَعْنَةٍ نَجلاءِ \n\n وغَمُوسٍ تَضِلُّ فيها يَدُ الآ   سِى ويَعيَى طبِيبُها بالدَواءِ \n\n رفعُوا رايةَ الضِرابِ وآلو   لَيَذُودُنّ سامِرَ المَلحاءِ""",
        """حَسَناتي عِندَ الزَمانِ ذُنوبُ   وَفَعالي مَذَمَّةٌ وَعُيوبُ \n\n وَنَصيبي مِنَ الحَبيبِ بِعادٌ   وَلِغَيري الدُنُوُّ مِنهُ نَصيبُ \n\n كُلُّ يَومٍ يُبري السُقامَ مُحِبٌّ   مِن حَبيبٍ وَما لِسُقمي طَبيبُ""",
        """فَكَأَنَّ الزَمانَ يَهوى حَبيباً   وَكَأَنّي عَلى الزَمانِ رَقيبُ \n\n إِنَّ طَيفَ الخَيالِ يا عَبلَ يَشفي   وَيُداوى بِهِ فُؤادي الكَئيبُ \n\n وَهَلاكي في الحُبِّ أَهوَنُ عِندي   مِن حَياتي إِذا جَفاني الحَبيبُ""",
        """مَنَعَ النَومَ ماوِيَ التَهمامُ   وَجَديرٌ بِالهَمِّ مَن لا يَنامُ \n\n مَن يَنَم لَيلَهُ فَقَد أُعمِلُ اللَي   لَ وَذو البَثِّ ساهِرٌ مُستَهامُ \n\n هَل تَرى مِن ظَعائِنٍ باكِراتٍ   كَالعَدَولِيِّ سَيرُهُنَّ اِنقِحامُ""",
        """حامِلُ الهَوى تَعِبُ   يَستَخِفُّهُ الطَرَبُ \n\n إِن بَكى يُحَقُّ لَهُ   لَيسَ ما بِهِ لَعِبُ \n\n تَضحَكينَ لاهِيَةً   وَالمُحِبُّ يَنتَحِبُ""",
        """حَفَّ كَأسَها الحَبَبُ   فَهيَ فِضَّةٌ ذَهَبُ \n\n أَو دَوائِرٌ دُرَرٌ   مائِجٌ بِها لَبَبُ \n\n أَو فَمُ الحَبيبِ جَلا   عَن جُمانِهِ الشَنَبُ""",
        """لا إله إلا الله   قولُ عارفٍ أوّاه \n\n أظهرت شهادته   حكمَ كلِّ من ناداه \n\n إن دعاه موجده   فالذي دعا لباه""",
        """تنطوي على أسفِ   يا خلي من الدنفِ \n\n قال للجفونِ أطع   تِ الهوى ولم تكفِ \n\n قد جنيت داهيةً   فاصبري أو اعترفي""",
        """مرَامُكم لا يُنالُ   كعثْرَة لا تُقالُ \n\n وذاكَ شيءٌ مُحالٌ  للسر منهُ مَجالُ \n\n نرى لكَ الدهْرَ مالا   يَسُوغُ منْهُ نوالُ""",
        """ولو بِقَدرِكَ أُهدِي   َمَا وَجدتُ هَدِيّه \n\n فَاقبَل بِفَضلِكَ نَزراً   قَبولُهُ كَالعَطِيّه""",
        """أَدنو إِليكَ فأُقْصَى   وكم أَطيعُ فأُعْصَى \n\n جَوْراً تَقَصَّيْتَ فيه   وجائِرٌ من تَقَصَّى \n\n عِشْقِي كمالٌ فما لِي   أَرَاهُ عِنْدَكَ نَقْصَا""",
        """حكوا لنا عن حمارٍ  من الحمير الغوالي \n\n أذناه أطول شيءٍ   ومخه جد خالي \n\n لكنه لا تضاعٍ   منه ورقة حال""",
        """سَيَندَمُ بَعضُكُم عَجلاً عليه  وما أَثرى اللِسانُ إِلى الجَرُوحِ \n\n فأنَّكم وما تُزجُونَ نحوي  مِنَ القَولِ المُرَغَّى والصَريحِ \n\n أَلا مَن مُبلِغُ الأَحلافِ عَنِّي  فَقَد تُهدَى النَصيحة للنصيحِ""",
        """فَتَردَعُه الدَّبُور لها أَجِيجٌ  ويُسلِمه إِلى الوَجدِ المَبِيتُ \n\n حِجَازِيُّ الهَوَى عَلِقٌ بِنَجدٍ  جَوِيٌّ لا يَعِيشُ ولا يَمُوتُ \n\n بَكَى فَرثَت له أَجبالُ صُبحٍ  وأسعَدتِ الجِبالَ بها مُرُوتُ""",
        """وأفلَتَنَا بَنُو شَكَلٍ رِجَالاً  حُفاةً يَربَؤُونَ عَلَى سُمَيرِ \n\n بَأنَّا قَد قَتَلنَا الخَيرَ قُرطاً  وَجُلنَا في سَراةِ بَني نُمَير \n\n أَلاَ أبلِغ بَني العَجلاَنِ عَنّي  فَلاَ يُنبِيكَ بِالحَدَثانِ غَيرِي""",
        """فَبَعضَ اللومِ عاذِلَتي فَإِنّي  سَتَكفيني التَجارِبُ وَاِنتِسابي \n\n عَصافيرٌ وَذِبّانٌ وَدودٌ  وَأَجرَأُ مِن مُجَلَّحَةِ الذِئابِ \n\n أَرانا موضِعينَ لِأَمرِ غَيبٍ  وَنُسحَرُ بِالطَعامِ وَبِالشَرابِ""",
        """قولا لِدودانَ عَبيدِ العَصا  ما غَرَّكُم بِالأَسَدِ الباسِلِ \n\n صُمَّ صَداها وَعَفا رَسمُها  وَاِستَعجَمَت عَن مَنطِقِ السائِلِ \n\n يا دارَ ماوِيَّةَ بِالحائِلِ  فَالسُهبِ فَالخَبتَينِ مِن عاقِلِ""",
        """باتوا يُصيبُ القَومُ ضَيفاً لَهُم  حَتّى إِذا ما لَيلُهُم أَظلَما \n\n إِذ قالَ عَمروٌ لِبَني مالِكٍ  لا تَعجَلوا المِرَّةَ أَن تُحكَما \n\n كانَ بَنو الأَبرَصِ أَقرانَكُم  فَأَدرَكوا الأَحدَثَ وَالأَقدَما""",
        """قُلتُ لِعَمروٍ حينَ أَرسَلتُهُ  وَقَد حَبا مِن دُونِهِ عالِجُ \n\n ولا قَعيدٌ أَغَضَبٌ قَرنُهُ  هاجَ لَهُ مِن مَرتَعٍ هائِجُ \n\n يا أَيُّها المُزمِعُ ثُمَّ اِنثَنى  لا يَثنِكَ الحازي وَلا الشاحِجُ""",
        """يَزخَرُ في أَقطارِهِ مُغدِقٌ  بِجَما فَتيِهِ الشوعَ وَالغِريَفِ \n\n مُعرَورِفٌ أَسبَلَ جُبَّارَهُ  أَسوَدُ كَالغابَةِ مُغدَودِفِ \n\n إِذا جَمادى مَنَعَت قِطرَها  زانَ جَناني عَطنٌ مُغضَفِ""",
        """حَسوداهُ في عُلاهُ  ظُبى البيض والقُطار \n\n اذا جادَ فهو غيْثٌ  واِن صالَ فهو نارُ \n\n وبالقصر أرْيحيٌّ  به يمنعُ الذِّمارُ""",
        """ولم يُصِبْنا سُروراً   وَلم يُلْهِنا سَماعا \n\n كأَنْ لم يَكُنْ جديراً  بِحِفْظِ الَّذي أَضاعا \n\n أَرى لِلصّبا وداعا  وما يَذْكُرُ اجْتماعا""",
        """يا بُدُوراً أنابَها الد  دَهْرََ عانٍ أسِيرُ \n\n طارَ قَلْبي بِحُبّها  مَنْ لِقَلْبٍ يطيرُ \n\n أَشْرَقَتْ لي بُدورُ  في ظلامٍ تُنيرُ""",
        """وَوَهبينُ في رُباها  لأَسرابِها رُبوعُ \n\n  رَعابيبُ مِن نُمَيرٍ  جَلابيبُها تَضوعُ \n\n بَدا لي عَلى الكَثيبِ  بِنَعمانَ ما يَروعُ""",
        """قلبي قلق جسمي شحب  ما بهما قسراً رقشا \n\n قولوا لهموا ما قصد هموا  وهواه لقلب الصب حشى \n\n الصبر ثوى لما حجبوا  بدراً قد أذاب هواه الحشى""",
        """يا نفسُ أحيَي تَصِلِي أمَلاً  عيشي رَجَباً تَرَي عَجَبا \n\n ذَرتِ السِّتونَ بُرادَتَها  فِي مِسكِ عِذارَكِ فاشتَهَبَا \n\n أَبعَيدَ الشّبابِ هَوى وَصِبا  كلا لا لَهوَ ولا لَعِبا""",
        """وعَنَت لعزائِمكم عربٌ  تشقى بصوارِمَها العجَمُ \n\n وهَمَت ديمٌ من راحَتِكُم  هيهاتَ تُساجِلُها الديَمُ \n\n شملَت ببقائِكِم النعَمُ  وسمتِ برجائِكُمُ الهِمَمُ""",
        """أَبِنَظرَةِ عَينٍ عَن خَطأٍ  عَرَضَت بِالعَمدِ يُراقُ دَمي \n\n فَتعالَي غَيرَ مُدافِعَةٍ  نَقصُص رُؤياكِ عَلى حَكَمِ \n\n مَن ذا أَفتاكِ بِسَفكِ دَمي  يا غُرَّةَ حَيِّ بَني جُشَمِ""",
    ],
    "Sea": [
        "بحر الطويل","بحر الطويل" ,"بحر الطويل", "بحر الطويل", "بحر الطويل",
        "بحر الرمل","بحر الرمل","بحر الرمل","بحر الرمل", "بحر الرمل",
        "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب",
        "بحر الكامل", "بحر الكامل", "بحر الكامل", "بحر الكامل",
        "بحر البسيط","بحر البسيط","بحر البسيط","بحر البسيط","بحر البسيط",
        "بحر المديد", "بحر المديد", "بحر المديد", "بحر المديد",
        "بحر الرجز", "بحر الرجز", "بحر الرجز", "بحر الرجز",
        "بحر الخفيف", "بحر الخفيف", "بحر الخفيف", "بحر الخفيف",
        "بحر المقتضب","بحر المقتضب", "بحر المقتضب", "بحر المقتضب",
        "بحر المجتث", "بحر المجتث","بحر المجتث","بحر المجتث",
        "بحر الوافر", "بحر الوافر","بحر الوافر","بحر الوافر",
        "بحر السريع", "بحر السريع", "بحر السريع", "بحر السريع",
        "بحر المضارع", "بحر المضارع", "بحر المضارع", "بحر المضارع",
        "بحر المتدارك", "بحر المتدارك", "بحر المتدارك", "بحر المتدارك"    
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

prompt = """مع الالتزام بقوانين التشكيل الشعرية التالية وفق بحر المعطى في السؤال وتفعيلاته، مع الحرص على أن تحمل معنى

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

def generate_poetry_response(query, threshold, model, enhance = False):
    results = arabic_VDB.similarity_search_with_score(query, k=2) # you can add k this is the number of the rag context
    context_text = "\n\n".join([doc.page_content for doc, score in results if score > threshold])
    input_with_rag = """{0}

لتوسيع مدارك فهمك يمكنك الإستلهام من هذه الأمثلة:
{1}

انشأ القصيدة مكونة من ثلاثة ابيات بناءً على هذا الطلب: {2} [/INST]""".format(prompt_result, context_text, query)
    response = model.generate(input_with_rag)['results'][0].get('generated_text')

    st.write("wihtout enhancment:")
    st.write(response)

    if enhance:
        enhanced_response = f"{input_with_rag}\n\n{response}</s>\n<s>[INST] احذف جميع الكلمات العربية [/INST]"

        responseX = model.generate(enhanced_response)['results']
        st.write(responseX)

        response = response[0].get('generated_text')

        st.write("with enhancment:")
        st.write(response)

    return response, context_text

# Streamlit App Start
st.title("أهلا بكم في ضيافة الشاعر النابغة المِلساني")

st.write("هنا تستطيع سؤال الشاعر ملسان عن أبيات او إنشاء قصائد من بحور متعددة من اختياركم")

# get API key
api_key = st.text_input("أدخل مفتاح الاستخدام")  

options = ["انشاء قصيدة", "اكمال قصيدة", "شرح قصيدة"]
selected_fruit = st.selectbox("اخر من خدمات ملسان", options)


prompt_result = """<s>[INST] <<SYS>>
انت شاعر عربي فصيح، عليك كتابة قصيدة شعرية باللغة العربية فقط

{0}

عند انشاء القصيدة يجب عليك كتابتها بناءً على هذه الشروط:
1) الإلتزام بالقواعد المذكورة اعلاه
2) انشاء قصيدتك الخاصة
3) القصيدة مُستلهمة بطريقة المُتنبي وليست بكلماته!
4) عدم ذكر التفعيلات بعد البيت ويجب استخدامها في بناء وزن البيت الشعري
5) لا يجب عليك استخدام الأمثلة المذكورة ووضعها في القصيدة انما يتم الإستلهام منها فقط!
<</SYS>>""".format(prompt)
    
# User Input
query = st.text_input("أكتب طلبك لإنشاء قصيدة من أحد البحور الشعرية ")
threshold = st.slider("أختر نسبة التقارب المطلوبة:", 0.0, 1.0, 0.9)

# Process Data and Display Results
if st.button("أطلق العنان"):
    status = st.empty()
	
    status.markdown('<div class="custom-text">يتم الإبداع...</div>', unsafe_allow_html=True)
    documents = create_documents(df) 
    arabic_VDB = create_embedding(documents)
	
    model_id = "sdaia/allam-1-13b-instruct"
    parameters = { 
        "decoding_method": "greedy", 
        "max_new_tokens": 400, 
        "repetition_penalty": 1 
	}

    model = Model(
        model_id = model_id,
        params = parameters,
        credentials = get_credentials(),
	    project_id = "11af8977-9294-4e73-a863-b7e37a214840",
    )

    if selected_fruit == "انشاء قصيدة" or selected_fruit == "اكمال قصيدة":
        response, rag = generate_poetry_response(query, threshold, model, enhance = True)
    else:
        response, rag = generate_poetry_response(query, threshold, model, enhance = False)
        
    status.write(f'<div class="custom-text">{response}</div>', unsafe_allow_html=True)
