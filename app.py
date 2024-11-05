import streamlit as st
import pandas as pd
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from ibm_watsonx_ai.foundation_models import Model
import markdown
import getpass


st.markdown(
    """
    <style>
    /* Hide the Streamlit top bar (header) */
    header { 
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# styling 
st.markdown(
    """
    <style>
    /* Set the background image */
    .stApp {
        background-image: url("https://raw.githubusercontent.com/salehtwj/Milsan/refs/heads/main/images/test.webp");  
        background-size: cover;
        background-position: center;
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

    </style>
    """,
    unsafe_allow_html=True
)

# Add logos fo the sponsers 
st.markdown(
    """
    <div class="bottom-left-logos">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/SDAIA_logo-removebg-preview.png?raw=true" alt="SDAIA Logo">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/safcsp_logo.png?raw=true" alt="SAFCSP Logo">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/tuwaiq_logo_w.png?raw=true/" alt="Tuwaiq Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Add Allam challenge logo
st.markdown(
    """
    <div class="top-left-logo">
        <img src="https://github.com/salehtwj/Milsan/blob/main/images/ALLaM_logo.png?raw=true" alt="ALLaM Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Load Data
data = {
    "Poem": [
        """قِفا نَبكِ مِن ذِكرى حَبيبٍ وَعِرفانِ   وَرَسمٍ عَفَت آياتُهُ مُنذُ أَزمانِ  أَتَت حُجَجٌ بَعدي عَلَيها فَأَصبَحَت   كَخَطِّ زَبورٍ في مَصاحِفِ رُهبانِ """
        ,""" أَعِنّي عَلى بَرقٍ أَراهُ وَميضِ  يُضيءُ حَبِيّاً في شَماريخَ بيضِ   وَيَهدَأُ تاراتٍ سَناهُ وَتارَةً   يَنوءُ كَتَعتابِ الكَسيرِ المَهيضِ """
        ,""" طَرِبتَ وَهاجَتكَ الظِباءُ السَوارِحُ  غَداةَ غَدَت مِنها سَنيحٌ وَبارِحُ  تَغالَت بِيَ الأَشواقُ حَتّى كَأَنَّما   بِزَندَينِ في جَوفي مِنَ الوَجدِ قادِحُ """
        ,""" أُعاتِبُ دَهراً لا يَلينُ لِعاتِبِ   وَأَطلُبُ أَمناً مِن صُروفِ النَوائِبِ   وَتوعِدُني الأَيّامُ وَعداً تَغُرُّني   وَأَعلَمُ حَقّاً أَنَّهُ وَعدُ كاذِبِ """
        ,""" دَهَتني صُروفُ الدَهرِ وَاِنتَشَبَ الغَدرُ  وَمَن ذا الَّذي في الناسِ يَصفو لَهُ الدَهرُ   وَكَم طَرَقَتني نَكبَةٌ بَعدَ نَكبَةٍ   فَفَرَّجتُها عَنّي وَما مَسَّني ضُرُّ"""

        ,

        """رُبَّ رامٍ مِن بَني ثُعَلٍ   مُتلِجٍ كَفَّيهِ في قُتَرِه   عارِضٍ زَوراءَ مِن نَشمٍ  غَيرُ باناةٍ عَلى وَتَرِه """
        ,""" نَفِّسوا كَربي وَداوُوا عِلَلي   وَاِبرِزوا لي كُلَّ لَيثٍ بَطَلِ   وَاِنهَلوا مِن حَدِّ سَيفي جُرَعاً   مُرَّةً مِثلَ نَقيعِ الحَنظَلِ """
        ,""" بَكَرَت تَعذُلُني وَسطَ الحِلالِ   سَفَهاً بِنتُ ثُوَيرِ بنِ هِلالِ   بَكَرَت تَعذُلُني في أَن رَأَت   إِبِلي نَهباً لِشَربٍ وَفِضالِ """
        ,""" ذادَ عَنى النَومَ هَمٌّ بَعدَ هَمّ   وَمِن الهَمِّ عَناءٌ وَسَقَم   طَرَقَت طَلحَةُ رَحلي بَعدَما  نامَ أَصحابى وَلَيلي لَم أَنَم """
        ,""" سَأَلَتني عَن أُناسٍ هَلَكوا  أَكَلَ الدَهرُ عَلَيهِم وَشَرِب"""

        ,

        """أَحارِ بنُ عَمروٍ كَأَنّي خَمِر  وَيَعدو عَلى المَرءِ ما يَأتَمِر  لا وَأَبيكَ اِبنَةَ العامِرِيِّ  لا يَدَّعي القَومُ أَنّي أَفِر """
        ,""" لَعَمرُكَ ما إِن لَهُ صَخرَةً  لَعَمرُكَ ما إِن لَهُ وَزَر """
        ,""" أَلَم تُكسَفِ الشَمسُ وَالبَدرُ وَالـ  ـكَواكِبُ لِلجَبَلِ الواجِبِ  لِفَقدِ فَضالَةَ لا تَستَوي الـ  ـفُقودُ وَلا خَلَّةُ الذاهِبِ """
        ,""" أَجِدّوا النِعالَ لِأَقدامِكُم  أَجِدّوا فَوَيهاً لَكُم جَروَلُ  وَأَبلِغ سَلامانَ إِن جِئتَها  فَلا يَكُ شِبهاً لَها المِغزَلُ """
        ,""" تُخَبِّرُني بِالنَجاةِ القَطاةُ  وَقَولُ الغُرابِ لَها شاهِدُ  تَقولُ أَلا قَد دَنا نازِحٌ  فِداءٌ لَهُ الطارِفُ التالِدُ"""

        ,

        """لِمَنِ الدِيارُ غَشِيتُها بِسُحامِ  فَعَمايَتَينِ فَهُضبُ ذي أَقدامِ  فَصَفا الأَطيطِ فَصاحَتَينِ فَغاضِرٍ  تَمشي النِعاجُ بِها مَعَ الآرامِ """
        ,""" هَل غادَرَ الشُعَراءُ مِن مُتَرَدَّمِ  أَم هَل عَرَفتَ الدارَ بَعدَ تَوَهُّمِ  يا دارَ عَبلَةَ بِالجَواءِ تَكَلَّمي  وَعَمي صَباحاً دارَ عَبلَةَ وَاِسلَمي """
        ,""" أَبَني زَبيبَةَ ما لِمُهرِكُمُ  مُتَخَدِّداً وَبُطونُكُم عُجرُ  أَلَكُم بِآلاءِ الوَشيجِ إِذا  مَرَّ الشِياهُ بِوَقعِهِ خُبرُ """
        ,""" قِف بِالمَنازِلِ إِن شَجَتكَ رُبوعُها  فَلَعَلَّ عَينَكَ تَستَهِلُّ دُموعُها  وَاِسأَل عَنِ الأَظعانِ أَينَ سَرَت بِها  آباؤُها وَمَتى يَكونُ رُجوعُها"""

        ,

        """كَم يُبعِدُ الدَهرُ مَن أَرجو أُقارِبُهُ  عَنّي وَيَبعَثُ شَيطاناً أُحارِبُهُ  فَيا لَهُ مِن زَمانٍ كُلَّما اِنصَرَفَت  صُروفُهُ فَتَكَت فينا عَواقِبُهُ """
        ,""" لا يَحمِلُ الحِقدَ مَن تَعلو بِهِ الرُتَبُ  وَلا يَنالُ العُلا مَن طَبعُهُ الغَضَبُ  وَمَن يِكُن عَبدَ قَومٍ لا يُخالِفُهُم  إِذا جَفوهُ وَيَستَرضي إِذا عَتَبوا """
        ,""" لَمّا جَفاني أَخِلّائي وَأَسلَمَني  دَهري وَلحمُ عِظامي اليَومَ يُعتَرَقُ  أَقبَلتُ نَحوَ أَبي قابوسَ أَمدَحُهُ  إِنَّ الثَناءَ لَهُ وَالحَمدُ يَتَّفِقُ """
        ,""" نفع قليلٌ إذا نادى الصدى أُصلا  وحانَ منه لبرد الماء تَغريد  وودعوني فقالوا ساعة انطلقوا  أودى فأودى النَدى والحزم والجود """
        ,""" قد أصبح الحبل من أسماء مصروما  بعد ائتلافٍ وحب كان مكتوما  واستبدلت خلة مني وقد علمت  أن لن أبيت بوادي الخسف مذموما"""
    ],
    "Sea": [
        "بحر الطويل", "بحر الطويل", "بحر الطويل", "بحر الطويل", "بحر الطويل",
        "بحر الرمل","بحر الرمل","بحر الرمل","بحر الرمل","بحر الرمل",
        "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب", "بحر المتقارب",
        "بحر الكامل", "بحر الكامل", "بحر الكامل", "بحر الكامل",
        "بحر البسيط","بحر البسيط","بحر البسيط","بحر البسيط","بحر البسيط"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame

# Streamlit App
st.title("اهلا بكم في ضيافة الشاعر النابغة الملساني")

st.write("هنا تستطيع سؤال الشاعر العظيم ملسان عن ابيات او انشاء قصائد من بحور متعددة من اختياركم")

# get API key
api_key = st.text_input("ادخل مفاتح الاستخدام")  

# User Input
query = st.text_input("اكتب طلبك سواء تقييم قصيدة معينه او انشاء قصيدة من احد البحور الشعرية ")
threshold = st.slider("Select similarity threshold:", 0.0, 1.0, 0.9)

# Define Functions
def load_data():
    return df

def create_documents(df):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100000)
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
        "apikey": api_key,  # Replace with your actual API key
        "project_id": "11af8977-9294-4e73-a863-b7e37a214840",  # Set your actual project_id here
        # Alternatively, use "space_id": "your_space_id_here" if applicable
    }


the_ofth_prompt = """
أنت شاعر عربي فصيح، عليك كتابة شطر من قصيدة باللغة العربية فقط، مع الالتزام بقوانين التشكيل الشعرية التالية وفق بحر المعطى في السؤال وتفعيلاته، مع الحرص على أن تحمل معنىً. إليك بعض القواعد وبعض الأمثلة لتساعدك في ذلك:

1. الحروف في اللغة العربية:

  1 الحرف الساكن: يمثل عدم وجود حركات، ويُشير إليه بعلامة السكون ( ْ ).
  2 الحرف المتحرك: يدل على اتجاه الصوت عند النطق، ويكون إما ضمة ( ُ ) أو فتحة ( َ ) أو كسرة ( ِ ).
  3 الشدة ( ّ ): تعبر عن تكرار الحرف، حيث يكون الأول ساكنًا والثاني متحركًا.

2. المقاطع العروضية:

1 السبب الخفيف:  يجمع حرفًا متحركًا ثم حرفًا ساكنًا مثل: لَمْ، عَنْ، كَمْ.
2 السبب الثقيل: ويكون عندما يجتمع حرفين متحركين مع بعضهما البعض مثل لَكَ - بِكَ - مَعَ.
3 الوتد المجموع: يتكون من حرفين متحركين وآخر ساكن مثل: إِلَى، عَلَى.
4 الوتد المفروق: يتكون من حرف متحرك، ثم ساكن، ثم متحرك مثل: أَيْنَ، قَاْمَ.
5 الفاصلة الصغرى: وهي تتألف من أربعة أحرف ، الثلاثة الأولى منها متحركة ، والرابع ساكن مثل لَعِبَتْ - فَرِجَتْ - رَجَعَاْ إلى آخره.
6 الفاصلة الكبرى: وهي تتألف من خمسة أحرف الأربعة الأولى متحركة والخامس ساكن مثل شَجَرَةٌ - ثَمَرَةٌ ( التنوين عبارة عن حركة يليها ساكن "شَجَرَتُنْ").


3. التفاعيل:

فَعُوْلُن: وهو يتألف من فعو + لن (وتد مجموع + سبب خفيف)
 فَاْعِلُنْ: وهو يتألف من فا + علن (سبب خفيف + وتد مجموع)
 مَفَاْعِيْلُنْ: وهو يتألف من مفا + عي + لن (وتد مجموع + سبب خفيف + سبب خفيف)
 مُفَاْعَلَتُنْ: وهو يتألف من مفا + علتن (وتد مجموع + فاصلة صغرى)
 مُتَفَاْعِلُنْ: وهو يتألف من متفا + علن (فاصلة صغرى + وتد مجموع)
 مَفْعُوْلَاْتُ: وهو يتألف من مف + عو + لات (سبب خفيف + سبب خفيف + وتد مفروق)
 مُسْتَفْعِلُنْ: وهو يتألف من مس + تف + علن (سبب خفيف + سبب خفيف + وتد مجموع) أو مس + تفع + لن (سبب خفيف + وتد مفروق + سبب خفيف)
  فَاْعِلَاْتُنْ: وهو يتألف من فا + علا + تن (سبب خفيف + وتد مجموع + سبب خفيف) أو فاع + لا + تن (وتد مفروق + سبب خفيف + سبب خفيف)

 بعد معرفة هذه المعلومات نبدأ ببحور الشعر ال16 عشر مع مقاطعه العروضية وتفعيلاته وترميزه

4. البحار:
  بحر الطويل: يجب أن يتبع أي شطر في قصائد بحر الطويل هذه التفعيلة " فَعُوْلُنْ مَفَاْعِيْلُنْ فَعُوْلُنْ مَفَاْعِيلُنْ
  بحر الرمل: يجب أن يتبع أي شطر في قصائد بحر الرمل هذه التفعيلة " فَاْعِلاتُنْ فَاْعِلاتُنْ فَاْعِلاتُنْ
  بحر المتقارب: يجب أن يتبع أي شطر في قصائد بحر المتقارب هذه التفعيلة "  فَعُوْلُنْ فَعُوْلُنْ فَعُوْلُنْ فَعُوْلُ
  بحر الكامل: يجب أن يتبع أي شطر في قصائد بحر الكامل هذه التفعيلة "  مُتَفَاْعِلُنْ مُتَفَاْعِلُنْ مُتَفَاْعِلُنْ
  بحر البسيط: يجب أن يتبع أي شطر في قصائد بحر البسيط هذه التفعيلة " مُسْتَفْعِلُنْ فَاْعِلُنْ مُسْتَفْعِلُنْ فَاْعِلُنْ

 !استخدم هذه الامثلة لتوسيع فهمك فقط!! يجيب عليك انشاء قصيدتك الخاصة

"""

def generate_poetry_response(query, threshold, model):
    results = arabic_VDB.similarity_search_with_score(query, k=4)
    context_text = "\n\n".join([doc.page_content for doc, score in results if score > threshold])
    input_with_rag = the_ofth_prompt + context_text + "\n\n" + "  اجب هنا بناء على هذا الطلب : " + query
    return model.generate(input_with_rag)['results'][0].get('generated_text') , context_text

# Process Data and Display Results
if st.button("Run Model"):
    st.write("Loading and processing documents...")
    documents = create_documents(df)
    arabic_VDB = create_embedding(documents)

    model_id = "sdaia/allam-1-13b-instruct"
    parameters = { "decoding_method": "greedy", "max_new_tokens": 200, "repetition_penalty": 1 }
    
    model = Model(
        model_id = model_id,
        params = parameters,
        credentials = get_credentials(),
        project_id = "11af8977-9294-4e73-a863-b7e37a214840"
	)
    response , rag = generate_poetry_response(query, threshold, model)
    st.write("Generated Poetry:")
    st.write(response)
    st.write("Generated RAG:")
    st.write(rag)



	


