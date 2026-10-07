# -*- coding: utf-8 -*-
"""DUR 성분명 → 약효군 분류. WHO INN stem + 명시 목록 병용."""
import re

def components(name):
    """복합제('a | b')를 분해하고 염·수화물·괄호 표기를 제거한다."""
    out = []
    for part in re.split(r'[|+/]', name or ''):
        p = part.lower().strip()
        p = re.sub(r'\([^)]*\)', ' ', p)                  # (as xxx), (5mg) 제거
        p = re.sub(r'\b\d+(\.\d+)?\s*(mg|g|ml|mcg|iu)\b', ' ', p)
        p = re.sub(r'\b(hydrochloride|hydrobromide|sulfate|sulphate|sodium|potassium|calcium|'
                   r'magnesium|maleate|tartrate|besylate|mesylate|citrate|acetate|succinate|'
                   r'fumarate|phosphate|nitrate|bitartrate|dihydrate|trihydrate|monohydrate|'
                   r'hydrate|anhydrous|micronized|microemulsion|diluted|l-proline|bis)\b', ' ', p)
        p = re.sub(r'\s+', ' ', p).strip()
        if p:
            out.append(p)
    return out or ['']

# --- 명시 목록 (stem으로 안 잡히거나 오분류되는 것) ---
EXPLICIT = {
    '조영제': ['iohexol','iopamidol','iomeprol','iopromide','ioversol','iodixanol','iobitridol',
              'ioxitalamate','iotrolan','ioxaglate','iotroxate','amidotrizoate','iodipamide',
              'iodised fatty acids','poppyseed oil','ferucarbotran','perflutren','sulfur hexafluoride'],
    '해열진통소염제': ['acetaminophen','paracetamol','aspirin','acetylsalicylic','ibuprofen','naproxen',
                 'diclofenac','aceclofenac','ketorolac','ketoprofen','dexibuprofen','zaltoprofen',
                 'loxoprofen','meloxicam','piroxicam','celecoxib','etoricoxib','polmacoxib',
                 'talniflumate','nabumetone','indomethacin','mefenamic'],
    '항생제': ['amoxicillin','ampicillin','cefaclor','cefixime','cefdinir','cefpodoxime','cephalexin',
             'cefditoren','ciprofloxacin','levofloxacin','moxifloxacin','ofloxacin','azithromycin',
             'clarithromycin','erythromycin','roxithromycin','doxycycline','minocycline','tetracycline',
             'clindamycin','metronidazole','trimethoprim','sulfamethoxazole','fosfomycin','nitrofurantoin',
             'rifampicin','rifabutin','isoniazid','ethambutol','pyrazinamide','linezolid','vancomycin'],
    '항진균제': ['itraconazole','fluconazole','terbinafine','griseofulvin','ketoconazole','voriconazole',
              'posaconazole','clotrimazole','miconazole'],
    '항히스타민제': ['loratadine','desloratadine','cetirizine','levocetirizine','fexofenadine','ebastine',
               'bepotastine','chlorpheniramine','hydroxyzine','ketotifen','bilastine','rupatadine'],
    '소화기': ['omeprazole','esomeprazole','lansoprazole','pantoprazole','rabeprazole','dexlansoprazole',
            'tegoprazan','famotidine','ranitidine','cimetidine','domperidone','metoclopramide',
            'mosapride','itopride','ondansetron','ramosetron','loperamide','bismuth','sucralfate',
            'ursodeoxycholic','simethicone','lactulose','bisacodyl'],
    '진해거담': ['dextromethorphan','codeine','ambroxol','acetylcysteine','carbocisteine','levodropropizine',
             'theophylline','doxofylline','montelukast','pseudoephedrine'],
    '진통마약성': ['tramadol','oxycodone','morphine','fentanyl','pethidine','hydromorphone','tapentadol','nalbuphine'],
    '근이완제': ['eperisone','afloqualone','baclofen','tizanidine','chlorzoxazone','methocarbamol','cyclobenzaprine'],
    '말라리아': ['mefloquine','hydroxychloroquine','chloroquine','primaquine','atovaquone','proguanil','artemether'],
    '백신': ['vaccine','hantaan virus','typhoid ty21a','bcg strain','varicella virus','varicella-zoster'],
    '피부': ['isotretinoin','acitretin','finasteride','dutasteride','minoxidil','tacalcitol','calcipotriol'],
    '항암제': ['methotrexate','palbociclib','ribociclib','abemaciclib','capecitabine','fluorouracil',
            'cisplatin','carboplatin','oxaliplatin','paclitaxel','docetaxel','doxorubicin','epirubicin',
            'cyclophosphamide','ifosfamide','etoposide','irinotecan','gemcitabine','pemetrexed',
            'imatinib','erlotinib','gefitinib','osimertinib','sunitinib','sorafenib','lenvatinib',
            'regorafenib','pazopanib','nilotinib','dasatinib','ponatinib','bosutinib','ruxolitinib',
            'ibrutinib','venetoclax','olaparib','niraparib','enzalutamide','abiraterone','bicalutamide',
            'tamoxifen','anastrozole','letrozole','exemestane','lenalidomide','thalidomide','pomalidomide',
            'bortezomib','carfilzomib','vincristine','vinblastine','vinorelbine','mitomycin','bleomycin',
            'cytarabine','azacitidine','decitabine','hydroxyurea','mercaptopurine','tioguanine','busulfan',
            'temozolomide','dacarbazine','procarbazine','tretinoin','arsenic trioxide','asparaginase'],
    '이식면역억제': ['cyclosporine','tacrolimus','sirolimus','everolimus','azathioprine','mycophenolate',
               'mizoribine','antithymocyte'],
    '항HIV': ['ritonavir','lopinavir','efavirenz','nevirapine','atazanavir','darunavir','dolutegravir',
            'raltegravir','emtricitabine','tenofovir','abacavir','lamivudine','zidovudine','cobicistat',
            'rilpivirine','bictegravir','maraviroc','saquinavir','indinavir','nelfinavir','tipranavir'],
    '파킨슨': ['levodopa','benserazide','carbidopa','rasagiline','selegiline','safinamide','pramipexole',
            'ropinirole','entacapone','amantadine','trihexyphenidyl','bromocriptine'],
    '폐동맥고혈압_발기부전': ['sildenafil','tadalafil','udenafil','vardenafil','mirodenafil','riociguat',
                    'bosentan','ambrisentan','macitentan','iloprost','treprostinil','selexipag'],
    '정신과': ['haloperidol','risperidone','olanzapine','quetiapine','aripiprazole','paliperidone',
            'ziprasidone','amisulpride','blonanserin','clozapine','chlorpromazine','sulpiride',
            'fluoxetine','paroxetine','sertraline','escitalopram','citalopram','fluvoxamine',
            'venlafaxine','desvenlafaxine','duloxetine','mirtazapine','bupropion','vortioxetine',
            'amitriptyline','nortriptyline','imipramine','clomipramine','trazodone','lithium',
            'alprazolam','lorazepam','diazepam','clonazepam','etizolam','zolpidem','triazolam',
            'buspirone','methylphenidate','atomoxetine','modafinil'],
    '항경련': ['carbamazepine','oxcarbazepine','valproate','valproic','lamotrigine','levetiracetam',
            'topiramate','phenytoin','phenobarbital','pregabalin','gabapentin','zonisamide','perampanel'],
    '내분비': ['metformin','glimepiride','gliclazide','glibenclamide','sitagliptin','linagliptin',
            'vildagliptin','saxagliptin','alogliptin','gemigliptin','evogliptin','teneligliptin',
            'dapagliflozin','empagliflozin','ipragliflozin','ertugliflozin','canagliflozin',
            'pioglitazone','lobeglitazone','rosiglitazone','insulin','liraglutide','dulaglutide',
            'semaglutide','levothyroxine','methimazole','propylthiouracil','prednisolone',
            'methylprednisolone','dexamethasone','hydrocortisone','deflazacort'],
    '순환기': ['amlodipine','nifedipine','felodipine','lercanidipine','benidipine','diltiazem','verapamil',
            'losartan','valsartan','telmisartan','candesartan','irbesartan','olmesartan','fimasartan',
            'ramipril','enalapril','lisinopril','perindopril','captopril','atenolol','bisoprolol',
            'carvedilol','metoprolol','propranolol','nebivolol','atorvastatin','rosuvastatin',
            'simvastatin','pravastatin','pitavastatin','fenofibrate','ezetimibe','clopidogrel',
            'ticagrelor','prasugrel','warfarin','rivaroxaban','apixaban','edoxaban','dabigatran',
            'digoxin','amiodarone','dronedarone','nitroglycerin','isosorbide','ivabradine',
            'furosemide','torasemide','spironolactone','hydrochlorothiazide','indapamide','sacubitril'],
    '비뇨기': ['tamsulosin','silodosin','alfuzosin','doxazosin','terazosin','solifenacin','tolterodine',
            'fesoterodine','mirabegron','oxybutynin','propiverine','imidafenacin'],
    '호흡기흡입': ['salbutamol','salmeterol','formoterol','indacaterol','vilanterol','budesonide',
              'fluticasone','beclometasone','ciclesonide','tiotropium','ipratropium','glycopyrronium',
              'umeclidinium','aclidinium'],
    '편두통': ['sumatriptan','zolmitriptan','naratriptan','rizatriptan','almotriptan','frovatriptan',
            'eletriptan','ergotamine','dihydroergotamine','flunarizine'],
}
# INN stem (접미사) — 명시 목록에 없을 때 적용
STEMS = [
    ('조영제',      [r'^io[a-z]+ol$', r'^gado']),
    ('항암제',      [r'ciclib$', r'tinib$', r'mab$', r'rubicin$', r'platin$', r'taxel$', r'zomib$',
                    r'parib$', r'lisib$', r'denib$', r'rafenib$']),
    ('항생제',      [r'cillin$', r'^cef', r'^ceph', r'floxacin$', r'thromycin$', r'cycline$', r'micin$']),
    ('항진균제',    [r'conazole$', r'fungin$']),
    ('항바이러스',  [r'ciclovir$', r'amivir$', r'buvir$', r'previr$', r'asvir$']),
    ('정신과',      [r'azepam$', r'zolam$', r'peridone$', r'piprazole$', r'setron$', r'oxetine$']),
    ('순환기',      [r'sartan$', r'pril$', r'dipine$', r'statin$', r'olol$', r'xaban$']),
    ('내분비',      [r'gliptin$', r'gliflozin$', r'glitazone$', r'glutide$']),
    ('해열진통소염제', [r'coxib$', r'profen$']),
    ('편두통',      [r'triptan$']),
]
# 군 의무대에서 실제 처방·투약되는 군 / 장병이 지참할 수 있는 만성질환 군 / 제외 군
MIL = {'해열진통소염제','항생제','항진균제','항히스타민제','소화기','진해거담','진통마약성',
       '근이완제','말라리아','백신','항바이러스'}
CHR = {'정신과','항경련','내분비','순환기','비뇨기','호흡기흡입','편두통','피부'}
EXCL = {'조영제','항암제','이식면역억제','항HIV','파킨슨','폐동맥고혈압_발기부전'}

def classify_one(comp):
    for cls, words in EXPLICIT.items():
        if any(w in comp for w in words):
            return cls
    for cls, pats in STEMS:
        if any(re.search(p, comp) for p in pats):
            return cls
    return '미분류'

def classify(name):
    return {classify_one(c) for c in components(name)}
