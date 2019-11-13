import os, torch
from transformers import BertTokenizer, BertForQuestionAnswering
from datetime import datetime

MODEL_PATH = 'D:/Github/trts_crawler/1.1/corpus server/trained_benchmark_case_100_cento'
PREDICTION_RESULT_PATH = MODEL_PATH + '/prediction_result.txt'
TOKEN_LIMITE = 512

DEADLINE_QUESTIONS = [
    'Qual o prazo?',
    'O autor deverá manifestar-se em quantos dias?',
]

def build_case_test(text, question_array):
    return {
        'text': text,
        'questions': question_array
    }

def build_deadline_case_test(text):
    return build_case_test(text, DEADLINE_QUESTIONS)


def answer(question, text):
    #input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
    #input_ids = tokenizer.encode(input_text)

    used_tokens = len(tokenizer.encode("[CLS] " + question + " [SEP] " + '' + " [SEP]"))
    remaining_tokens = token_limit - used_tokens
    text_ids = tokenizer.encode(text)[-remaining_tokens:]
    input_ids = tokenizer.encode("[CLS] " + question + " [SEP] ") + text_ids + tokenizer.encode(" [SEP]")

    '''
    # Predict hidden states features for each layer
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs[0]
    '''

    with torch.no_grad():

        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]).to('cuda'), token_type_ids=torch.tensor([token_type_ids]).to('cuda'))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace(" ##", "")

CASES_TEST = [
    build_case_test(('No entanto, o valor requerido na inicial figura elevado diante dos elementos trazidos, de modo que fixo em R$8.000,00 '
    '(oito mil reais) a indenização a esse título, que entendo suficiente para amenizar os transtornos causados ao autor pela conduta da ré, sem configurar causa de '
    'enriquecimento indevido e, por outro lado, para incentivar a ré a adotar práticas comerciais mais eficientes, de modo a não causar transtornos e prejuízos indevidos a '
    'seus clientes, bem como a solucionar no âmbito administrativo de forma rápida e eficaz, eventuais problemas surgidos na rotina de suas práticas negociais e tratar seus '
    'clientes com mais respeito. Ante o exposto, nos termos do artigo 487, I, do Código de Processo Civil, JULGO PARCIALMENTE PROCEDENTE o pedido deduzido por '
    'VALDINEZ RIBEIRO DE CASTRO contra VRG LINHAS AÉREAS S/A. CONDENO a ré a pagar ao autor a quantia de R$8.000,00 (oito mil reais), a título de indenização por danos morais, '
    'corrigida conforme Súmula 362 do STJ e acrescida de juros moratórios de 1% a partir desta data. Julgo extinto o processo, nos termos do artigo 485, VI, do Código de '
    'Processo Civil em relação à ré Decolar.com Ltda. Sem sucumbência por força do disposto no artigo 55 da Lei n. 9.099/95. P. R. I. C. São Paulo, 28 de agosto de 2017. '
    'ANA CRISTINA WEYNEN CORES DEPIERIJuíza de Direito - ADV: GUSTAVO ANTONIO FERES PAIXÃO (OAB 186458/SP), MARILIA MICKEL MIYAMOTO NALETTO TEIXEIRA (OAB 271431/SP), '
    'MÁRCIA POSZTOS MEIRA PLATES (OAB 350159/SP)'),
    [
        'Qual o valor da condenação?'
    ]),

    build_case_test(('INSTADO A MANIFEST AR-SE, O EXEQUENTE ARGUMENTOU QUE OCORREU FRAUDE A EXECUCAO E REQ UEREU A DECLARACAO DE INEFICACIA DA ALIENACAO DO VEICULO, FLS.52/ 53. '
    'DESPACHO DE FL.56 DETERMINOU A INTIMACAO DO EXECUTADO, NAO SE NDO ESTE ENCONTRADO NO ENDERECO INFORMADO NOS AUTOS, CONFORME AVI SO DE RECEBIMENTO DE FL.59. '
    'EM PETICAO DE FLS.81/82 A EXEQUENTE R EQUEREU A REALIZACAO DE DILIGENCIAS JUNTO AO SISTEMA BACENJUD, A FIM DE LOCALIZAR O ENDERECO ATUALIZADO DO EXECUTADO. '
    'CUMPRE SALIE NTAR QUE E DEVER DA PARTE MANTER SEU ENDERECO ATUALIZADO, DEVENDO ARCAR COM O ONUS DE SUA OMISSAO. ASSIM, A PARTIR DO MOMENTO EM Q UE O EXECUTADO FOI '
    'CITADO, CABERIA A ESTE MANTER ESTE JUIZO INFOR MADO DE SEU ENDERECO. NESSE PASSO, DETERMINO A EXPEDICAO DE MANDA DO DE INTIMACAO PARA O EXECUTADO, A SER CUMPRIDO NO '
    'ENDERECO INFO RMADO NOS AUTOS, A FIM DE DAR CUMPRIMENTO AO DESPACHO DE FL.56. D ILIGENCIE-SE JUNTO AO DETRAN/GO, EXPEDINDO-SE OFICIO CASO SE FACA NECESSARIO, A FIM DE '
    'OBTER INFORMACAO A RESPEITO DA TRANSFERENCI A DO VEICULO GM/S10 EXECUTIVE D, ANO 2008, PLACA NLU 0390. COM A RESPOSTA, INTIME-SE O EXEQUENTE PARA MANIFESTAR-SE NO PRAZO '
    'DE CI NCO DIAS, DEVENDO INFORMAR O ENDERECO DO TERCEIRO ADQUIRENTE, A F IM DE QUE O MESMO SEJA INTIMADO, CONFORME DETERMINA O ART.792, 4, DO CPC. INTIMEM-SE. PIRES DO '
    'RIO, ___/___/2017. HELIO ANTONIO CR ISOSTOMO DE CASTRO JUIZ DE DIREITO'),
    [
        'Houve intimação pelo juiz?',
        'O que foi determinado?',
        'O que foi intimado?',
    ]),

    build_case_test(('Processo Nº RTOrd-0000872-09.2017.5.20.0011 AUTOR RODRIGO DOS SANTOS ADVOGADO ADALICIO MORBECK NASCIMENTO JUNIOR(OAB: 4379/SE) RÉU MAPSOLO ENGENHARIA LTDA EM RECUPERACAO '
    'JUDICIAL RÉU PETROLEO BRASILEIRO S A PETROBRAS Intimado(s)/Citado(s): - RODRIGO DOS SANTOS PODER JUDICIÁRIO JUSTIÇA DO TRABALHO DECISÃO - PJe Vistos e etc. Indefiro, por '
    'ora, o pleito de antecipação de tutela, deixando para apreciá-lo quando da realização da sessão já designada. Notifiquem-se as reclamadas para comparecimento à audiência, '
    'com as advertências de praxe. MARUIM, 14 de Agosto de 2017 CRISTIANE D AVILA RIBEIRO Juiz do Trabalho Titular'),
    [
        'Foi defirido ou indefirido?',
        'Qual a decisão da tutela?',
        'Qual o resultado da publicação?',
        'Qual o deferimento?',
        'Quem é o autor do processo?',
        'Quem é o réu?',
        'O que é o processo?',
        'Qual o resultado da audiência?',
        'Qual o resultado do pedido?',
        'O que foi notificado',
        'Quem foi notificado?',
        'Houve notificação para audiência?',
    ]),

    build_case_test(('INTIMAÇÃO EFETIVADA REF. À MOV. Sentença Extinto o Processo Sem Resolução do Mérito - 03/08/2017 13:47:21 LOCAL : ANÁPOLIS - 2ª VARA CÍVEL NR. PROCESSO : '
    '5247034.62.2017.8.09.0006 CLASSE PROCESSUAL : Tutela Cautelar Antecedente POLO ATIVO : ELIABE RORIZ SILVA POLO PASSIVO : ASSOCIAÇÃO EDUCATIVA EVANGÉLICA '
    '(CENTRO UNIVERSITÁRIO DE ANÁPOLIS UNIEVANGÉLICA) SEGREDO JUSTIÇA : NÃO PARTE INTIMADA : ELIABE RORIZ SILVA ADVGS. PARTE : 45192 GO - EVELLYN LESSA GONÇALVES DOS SANTOS '
    '47164 GO - GABRIEL RODRIGUES DE OLIVEIRA - VIDE ABAIXO O(S) ARQUIVO(S) DA INTIMAÇÃO. Anápolis - 2ª Vara Cível 5247034.62.2017.8.09.0006 ELIABE RORIZ SILVA ASSOCIAÇÃO '
    'EDUCATIVA EVANGÉLICA (CENTRO UNIVERSITÁRIO DE ANÁPOLIS ? UNIEVANGÉLICA) HOMOLOGO, para que surtam seus jurídicos e legais efeitos, a desistência da ação (evento nº 4) e, '
    'por consequência, julgo extinto o processo, sem resolução de mérito, nos termos do art. 485, VIII, CPC, desnecessária a oitiva da parte contrária, posto que, embora tenha '
    'inserido contestação (evento 5), o pedido de desistência foi formulado anteriormente, quando sequer havia despacho de recebimento da inicial e determinação para citação. '
    'Defiro o pedido de Gratuidade da Justiça (art. 99, § 3º, CPC). Sem custas e honorários. Transitada em julgado, arquivem-se. Anápolis, 3 de agosto de 2017 Algomiro '
    'Carvalho Neto Juiz de Direito NR. PROCESSO: 5247034.62.2017.8.09.0006'),
    [
        'Numero do processo?',
        'Qual é o processo?',
        'Qual a vara?',
        'Foi homologado ou não homologado?',
        'Qual o evento de contestação?',
        'Quando o pedido de desistência foi formulado?',
        'Qual a cidade do processo?',
        'Qual a cidade?',
        'O que ocorreu com a ação?',
        'Qual o julgamento?',
        'Qual a decisão?',
        'O que foi julgado?',
        'O que foi julgado do processo?',
        'O que foi julgado no processo?',
        'Houve resolução de mérito?',
        'Quais os termos?',
        'Quais os termos do artigo?',
        'Quais os termos da fundamentação?',
        'Qual o resultado?',
        'A oitiva da parte contrária é necessária?',
    ]),

    build_case_test(('NR. PROTOCOLO : 594184-06.2008.8.09.0026 ( 200805941848 ) AUTOS NR. : 277 NATUREZA : ACAO PENAL VITIMA : MARCOS SILVA LIMA ACUSADO : ANDRE DOS SANTOS NEVES ADV VIT : '
    '12359 GO - JONAS LEONARDO COSTA BARBOSA ADV ACUS : 9783 GO - NILSON NUNES REGES DESPACHO : PROTOCOLO : 200805941848 DESPACHO COMPULSANDO DETIDAMENTE O PRESE NTE FEITO, '
    'PERCEBE-SE ATRAVES DA DECISAO DE FLS. 581/584, QUE JA HOUVE ANALISE DE UM RECURSO DE APELACAO INTERPOSTO PELO ACUSADO L UIZ ANDRE RODRIGUES DE SOUZA (FLS. 568/580) O '
    'QUAL NAO FOI CONHEC IDO EM RAZAO DA INTEMPESTIVIDADE. DIANTE DO EXPOSTO, NAO CONHECO DO RECURSO DE APELACAO INTERPOSTO PELA DEFESA AS FL. 641/648, POR QUANTO NAO '
    'ATENDIDO O PRESSUPOSTO PROCESSUAL RECURSAL REFERENTE A TEMPESTIVIDADE. INTIMEM-SE. DE OUTRO LADO, UMA VEZ QUE JA OCORRE U O TRANSITO EM JULGADO DA SENTENCA EM RELACAO AO '
    'ACUSADO LUIZ AN DRE RODRIGUES DE SOUZA (CERTIDAO DE FL. 584-VERSO), BEM COMO JA H OUVE A EXPEDICAO DA GUIA DE EXECUCAO PENAL, DETERMINO QUE SEJA DA DO BAIXA DO SISTEMA E '
    'NA CAPA DOS AUTOS, PASSANDO O PRESENTE FEIT O A TRAMITAR SOMENTE EM RELACAO AO ACUSADO ANDRE DOS SANTOS NEVES . SEM PREJUIZO DA DILIGENCIA ACIMA, CUMPRA-SE NA INTEGRA A '
    'DECISA O DE FL. 640. CAMPOS BELOS/GO, 14 DE JUNHO DE 2017. FERNANDO MARN EY OLIVEIRA DE CARVALHO JUIZ SUBSTITUTO (DECRETO JUDICIARIO N 2.0 22 /2016 )'),
    [
        'Houve expedição da guia?',
        'Houve prejuízo da diligência?',
        'Qual o julgamento?',
        'Qual a decisão?',
        'Será cumprido a decisão de qual folha?',
        'Qual o número do decreto?',
        'Houve análise de recurso?',
        'Quem é o acusado?',
        'Quem é a vítima?',
        'Qual o recurso?',
        'O que foi determinado?',
        'Qual o pressuposto processual?',
        'O recurso de apelação foi conhecido?',
        'O pressuposto foi atendido?',
        'O pressuposto processual recursal foi atendido?',
        'A decisão será cumprida?',
        'O que será cumprido?',
        'Houve análise de recurso de apelação?',
        'Houve trânsito em julgado?',
        'trânsito em julgado',
        'Ocorreu trânsito em julgado?',
    ]),

    build_case_test(('PROCESSO: 00107282120148140301 PROCESSO ANTIGO: ---- MAGISTRADO(A)/RELATOR(A)/SERVENTUÁRIO(A): CESAR AUGUSTO PUTY PAIVA RODRIGUES Ação: Procedimento Comum em: '
    '26/07/2017 AUTOR:RAUL COSTA VELOSO Representante(s): OAB 17570 - ARIADNE OLIVEIRA MOTA DURANS (ADVOGADO) REU:BANCO PSA FINANCE BRASIL SA Representante(s): OAB 20599-A - '
    'MARCO ANDRE HONDA FLORES (ADVOGADO) . SENTENÇA Cuida-se de Ação Revisional de Contrato c/c Consignação em Pagamento, ajuizada por RAUL COSTA VELOSO em desfavor de BANCO '
    'PSA FINANCE BRASIL S. A., já devidamente qualificados nos autos em epígrafe. Às fls. 94-96, as partes peticionaram conjuntamente um termo de acordo, informando a '
    'composição amigável realizada extrajudicialmente entre ambas e requerendo a homologação do pacto com a conseqüente extinção da ação, nos termos do Art. 487, III, \'b\', '
    'do CPC/2015. É o que merece relato. Decido. Tendo sido observadas as formalidades legais, HOMOLOGO POR SENTENÇA o acordo formulado pelas partes (fls. 94-96) e inclusive '
    'já cumprido, para que produza seus efeitos legais e jurídicos. Por corolário, JULGO EXTINTO o processo com resolução do mérito, nos termos do art. 487, III, \'b\', do '
    'CPC/2015. Sem custas adicionais, nos termos do Art. 90, §3º, do CPC/2015. Após, as cautelas legais e de praxe, ARQUIVE-SE. Intime-se. Belém-PA, 19 de julho de 2017. '
    'CÉSAR AUGUSTO PUTY PAIVA RODRIGUES Juiz de Direito da 11ª Vara Cível e Empresarial de Belém'),
    [
        'O processo têm custas adicionais?',
        'O que foi decidido?',
        'Qual o julgamento?',
        'Cuida-se do que?',
        'Quem ajuizou?',
        'Quem homologou?',
        'Qual a homologação?',
        'O que as partes peticionaram?',
        'O que o termo de acordo informou?',
        'O que ambas requerem?',
        'Requerem qual homologação?',
        'Qual a extinção?',
        'Como foi homologado?',
        'Qual acordo foi homologado?',
        'Qual a vara do Juiz?',
        'Qual o juiz?',
        'Qual o juiz de direito?',
        'Quem são os representantes?',
        'Quem é o réu?',
        'Qual a sentença?',
        'Qual o número do processo?',
        'Qual o processo?',
        'O que foi julgado?',
        'O que foi julgado extinto?',
        'Foi julgado extinto?',
        'Foi julgado extinto ou procedente?',
        'O processo foi julgado extinto ou procedente?',
        'Como o processo foi julgado?',
    ]),

    build_case_test(('Eduardo Jesus de Almeida ajuizou a presente Ação de Cobrança de Indenização Securitária em face de Bradesco Previdência e Seguros S/A e Mapfre Vida S. A., todos '
    'qualificados e representados nos autos. Reconhecida a conexão do presente feito com o processo de cód. 847232, o feito foi redistribuído para este Juízo. As partes '
    'informam que celebraram acordo referente ao presente feito e o processo de cód. 847232, pugnam pela sua homologação e extinção dos processos. Relatado o necessário. '
    'Decido. Apensei o presente feito ao processo de cód. 847232, em razão da existência de conexão. Anote junto ao sistema Apolo o apensamento dos autos. Observo que o '
    'acordo firmado entre as partes (fls. 381/verso) versam sobre direitos disponíveis, de modo que o homologo por sentença para que produza seus legais e jurídicos '
    'efeitos. Diante do exposto, tendo a conciliação efeito de sentença entre as partes, JULGO EXTINTO o processo com resolução de mérito, nos termos do artigo 487, inciso '
    'III, alínea \'b\'\' do Código de Processo Civil. Honorários advocatícios conforme o acordo. Não há que se falar em custas remanescentes, em razão do disposto no artigo '
    '90, §3º, do Código de Processo Civil. Considerando que as partes desistiram do prazo recursal, arquivem-se os autos com as baixas e anotações necessárias. Publique-se. '
    'Intimem-se. Cumpra-se.'),
    [
        'Como o processo foi julgado?',
        'Quem ajuizou a presente ação?',
        'O processo foi julgado extinto com base em quais termos?',
        'O processo foi julgado extinto com quais termos?',
    ]),

    build_case_test(('A maioria das diferenças nos números revisados de déficit orçamentário deveu-se a uma mudança temporária das práticas contábeis do novo governo, ou seja, '
    'registrando despesas quando o material militar foi encomendado e não recebido. Contudo, foi a aplicação retroativa da metodologia SEC95 (aplicada desde 2000) pelo '
    'Eurostat, que finalmente elevou o déficit orçamentário do ano de referência (1999) para 3,38% do PIB, excedendo, assim, o limite de 3%. Isso levou a alegações de que '
    'a Grécia (alegações semelhantes foram feitas sobre outros países europeus como a Itália) não cumpriram todos os cinco critérios de adesão e a percepção comum de que a '
    'Grécia entrou na zona do euro por meio de números de déficit \"falsificados\".'),
    [
        'Quando as despesas foram registradas pelo novo governo?',
        'Desde quando é aplicado?',
        'Os cinco critérios de adesão foram cumpridos?',
    ]),

    build_case_test(('Trata-se de agravo interno manejado por João Rodrigues Itaboray desafiando decisão que extinguiu o mandado de segurança, sem apreciação do mérito, nos termos do art. '
    '485, V, do Código de Processo Civil/2015 (litispendência). Alega o agravante que (e-STJ, fl. 309): O presente Mandado de Segurança foi denegado em decorrência de '
    'suposta litispendência baseada em informação da Autoridade Coatora, a qual alegou que o Impetrante ajuizou a Execução de Título Extrajudicial n. '
    '0032554-51.2013.4.01.3400 perante a 19ª Vara Federal da Seção Judiciária do DF. Ocorre que a Autoridade Coatora omitiu, dolosamente, o fato de que o citado processo '
    'de Execução foi objeto de Embargos à Execução de n. 0059738-79.2013.4.01.3400, tendo este sido sentenciado de forma favorável à União em 20/01/2017, no sentido '
    'de extinguir o processo de execução por ausência de título executivo. Argumenta, assim, que, diante da extinção dos embargos à execução, cujo trânsito em julgado '
    'ocorrera em 2/3/2017, não se pode mais falar em litispendência, não existindo, por isso, nenhum óbice para o perfeito processamento do presente mandado de segurança.'),
    [
        'Trata-se do que?',
        'Por quem foi manejado?',
        'O que foi extinguido pela decisão?',
        'Houve apreciação do mérito?',
        'O que se argumenta?',
        'É possível falar em litispendência?',
        'Quando ocorreu o trânsito em julgado?',
        'Existe algum óbice para o processamento?',
        'Trata-se de qual ação?',
        'Quem manejou o agravo interno?',
        'Quem manejou a ação?',
        'O mandado de segurança foi denegado ou concedido?',
        'O presente mandado de segurança foi denegado ou concedido?',
        'Qual o artigo do código de processo civil?',
        'Qual a alegação do agravante?',
        #'Qual a alegação do impetrante?',
        'Qual a alegação da autoridade coatora?',
        'O que a autoridade coatora alegou',
        'O que a autoridade coatora alegou?',
    ]),

    build_case_test(('Trata-se de mandado de segurança impetrado por João Rodrigues Itaboray contra ato omissivo supostamente praticado pelo Ministro de Estado da Defesa, consistente no '
    'não pagamento do valor determinado na Portaria n. 1.009/2005, que seria devido como efeitos retroativos de reparação econômica em decorrência do reconhecimento da '
    'condição de anistiado político. Busca o impetrante a percepção do valor de R$ 214.379,23 (duzentos e quatorze mil, trezentos e setenta e nove reais e vinte e '
    'três centavos), alusivo aos efeitos financeiros retroativos, no que tange ao período compreendido entre 3/8/1999 e 3/3/2005, data do julgamento pela Comissão de '
    'Anistia. Aduz que o direito líquido e certo que fundamenta o presente writ foi definido pelo Supremo Tribunal Federal no RE 553.710 (Tema 394 da Repercussão Geral), o '
    'qual afastou as preliminares de decadência e de descabimento do mandado de segurança, decidindo pela determinação à União de imediato pagamento dos valores '
    'retroativos previstos nas respectivas portarias anistiadoras.'),
    [
        'Qual a percepção de valor?',
        'Qual o período compreendido?',
        'O que ocorreu com as preliminares de decadência?',
        'Quem determinou o pagamento dos valores retroativos?',
        #'A quem foi determinado o pagamento dos valores retroativos?',
        'Quem foi determinado para o pagamento dos valores retroativos?',
        'Qual a determinação?',
        'Qual a data do julgamento?',
        'Qual o tema de repercussão geral?',
        'Trata-se de qual ação?',
        'Qual a portaria?',
    ]),

    build_case_test(('Requer, ao final, a reconsideração da decisão recorrida, para o fito de dar prosseguimento ao mandado de segurança, até final julgamento. Intimada para que se '
    'manifestasse, a ora agravada deixou de apresentar impugnação, conforme atesta a certidão de e-STJ, fl. 325. É o relatório. Entendo que assiste razão à parte '
    'agravante. Com efeito, extinta, na origem, a Execução de Título Extrajudicial n. 0032554-51.2013.4.01.3400, a qual buscava o pagamento dos valores retroativos '
    'fixados na Portaria n. 1.009/2005, que reconheceu a condição de anistiado político ao impetrante, não há que se cogitar de litispendência entre o referido processo '
    'e a pretensão deduzida no presente writ. Sendo assim, reconsidero a decisão agravada para afastar a litispendência na hipótese e passo ao julgamento de mérito do '
    'presente mandamus.'),
    [
        'O que é requerido?',
        'Quem foi intimada para que se manifestasse?',
        'O que a ora agravada fez?',
        'Entendo que assiste o que?',
        'Buscava o pagamento de quais valores',
        'Reconheceu que condição?',
        'Há que se cogitar de litispendência?',
        'O que se é requerido?',
    ]),

    build_case_test(('No mérito, pugna pela denegação da ordem, porque: (a) recomendações expressas do TCU e da AGU impossibilitam o pagamento; (b) não há disponibilidade orçamentária '
    'específica; (c) deve ser respeitado o princípio da reserva do possível; (d) não podem incidir juros e correção monetária sobre o valor fixado na portaria; e (e) a '
    'concessão da segurança provocaria desequilíbrio no orçamento da União. O Ministério Público Federal manifestou-se pela parcial concessão da ordem, nos termos do parecer '
    'de e-STJ, fls. 279/285. Como relatado, a questão controvertida diz respeito à possibilidade de o impetrante, na condição de militar anistiado, perceber reparação '
    'econômica retroativa, em parcela única, que não foi paga diante da inércia do impetrado. Inicialmente, imperioso rechaçar a prejudicial de decadência. Consoante '
    'cediço, a impetração de mandamus contra ato omissivo de natureza continuada, como ocorre no descumprimento de determinação de pagamento de reparação econômica em '
    'prestação mensal, permanente e continuada, com efeitos retroativos (Lei n. 10.559/2002), não se subsume aos efeitos da decadência, conforme a hodierna jurisprudência '
    'desta Corte e do Pretório Excelso.'),
    [
        'Como o MPF se manifestou?',
        'O MPF se manifestou com quais termos?',
        'O MPF se manifestou em quais termos?',
        'A questão controvertida diz respeito ao que?',
        'Se subsume aos efeitos da decadência?',
        'Há disponibilidade orçamentária específica?',
        'Deve ser respeitado o princípio da reserva do possível?',
        'O que a concessão da segurança provocaria?',
    ]),

    build_case_test(('ANTE O EXPOSTO, homologo o acordo firmado pelas partes, para que produza seus jurídicos efeitos e JULGO EXTINTO o processo, com apreciação do mérito, na forma do '
    'artigo 487, inciso III, \'b\', do Novo Código de Processo Civil. Custas finais conforme acordado. Faculto o desentranhamento dos documentos mediante traslado. Transitado '
    'em julgado, arquivem-se os autos com baixa. P. R. I. Núcleo Bandeirante - DF, quarta-feira, 05/07/2017 às 15h17. Magáli Dellape Gomes,Juíza de Direito .'),
    [
        'O que julgo?',
        'O que será feito com os autos?',
        'O acordo entre as partes será homologado?',
        'Na forma de qual artigo?',
        'Qual artigo do código de processo civil?',
    ]),
    
    build_deadline_case_test(('Numeração única: 28139-54.2015.4.01.3400\n28139-54.2015.4.01.3400 AÇÃO ORDINÁRIA / OUTRAS\nAUTOR :   ITAU UNIBANCO S.A.\nADVOGADO :   '
    'SP00198407 - DIOGO PAIVA MAGALHAES VENTURA\nREU :   UNIAO FEDERAL\nO Exmo. Sr. Juiz exarou :\n\" Vista ao autor para manifestar-se no prazo de 10 (dez) dias acerca dos '
    'embargos\ndeclaratórios opostos pela União, fls. 240/242, tendo em vista seus possíveis efeitos\nmodificativos.\"')),

    build_deadline_case_test(('AS FLS.44/46 O EXEQUENTE REQUEREU A PENHORA SOBRE OS\nDIREITOS DO EXECUTADO INCIDENTES SOBRE O VEICULO LOCALIZADO. EM\nNOVA PESQUISA ATRAVES DO '
    'SISTEMA RENAJUD, CONSTATOU-SE QUE O VEIC\nULO FOI ALIENADO PARA RAFAEL THOMAZINI, FL.48. INSTADO A MANIFEST\nAR-SE, O EXEQUENTE ARGUMENTOU QUE OCORREU FRAUDE A EXECUCAO '
    'E REQ\nUEREU A DECLARACAO DE INEFICACIA DA ALIENACAO DO VEICULO, FLS.52/\n53. DESPACHO DE FL.56 DETERMINOU A INTIMACAO DO EXECUTADO, NAO SE\nNDO ESTE ENCONTRADO NO '
    'ENDERECO INFORMADO NOS AUTOS, CONFORME AVI\nSO DE RECEBIMENTO DE FL.59. EM PETICAO DE FLS.81/82 A EXEQUENTE R\nEQUEREU A REALIZACAO DE DILIGENCIAS JUNTO AO SISTEMA '
    'BACENJUD, A\nFIM DE LOCALIZAR O ENDERECO ATUALIZADO DO EXECUTADO. CUMPRE SALIE\nNTAR QUE E DEVER DA PARTE MANTER SEU ENDERECO ATUALIZADO, DEVENDO\nARCAR COM O ONUS DE '
    'SUA OMISSAO. ASSIM, A PARTIR DO MOMENTO EM Q\nUE O EXECUTADO FOI CITADO, CABERIA A ESTE MANTER ESTE JUIZO INFOR\nMADO DE SEU ENDERECO. NESSE PASSO, DETERMINO A EXPEDICAO '
    'DE MANDA\nDO DE INTIMACAO PARA O EXECUTADO, A SER CUMPRIDO NO ENDERECO INFO\nRMADO NOS AUTOS, A FIM DE DAR CUMPRIMENTO AO DESPACHO DE FL.56. D\nILIGENCIE-SE JUNTO AO '
    'DETRAN/GO, EXPEDINDO-SE OFICIO CASO SE FACA\nNECESSARIO, A FIM DE OBTER INFORMACAO A RESPEITO DA TRANSFERENCI\nA DO VEICULO GM/S10 EXECUTIVE D, ANO 2008, PLACA NLU '
    '0390. COM A\nRESPOSTA, INTIME-SE O EXEQUENTE PARA MANIFESTAR-SE NO PRAZO DE CI\nNCO DIAS, DEVENDO INFORMAR O ENDERECO DO TERCEIRO ADQUIRENTE, A F\nIM DE QUE O MESMO '
    'SEJA INTIMADO, CONFORME DETERMINA O ART.792, 4,\nDO CPC')),

    #####

    build_deadline_case_test(('Numeração única: 28139-54.2015.4.01.3400 28139-54.2015.4.01.3400 AÇÃO ORDINÁRIA / OUTRAS AUTOR : ITAU UNIBANCO S. A. ADVOGADO : SP00198407 - '
    'DIOGO PAIVA MAGALHAES VENTURA REU : UNIAO FEDERAL O Exmo. Sr. Juiz exarou : \' Vista ao autor para manifestar-se no prazo de 10 (dez) dias acerca dos embargos '
    'declaratórios opostos pela União, fls. 240/242, tendo em vista seus possíveis efeitos modificativos.\'')),

    build_deadline_case_test(('Processo 0302662-06.2009.8.12.0011 (011.09.302662-6) - Execução de Título Extrajudicial - Causas Supervenientes à Sentença Exeqte: Banco '
    'Bradesco S/A ADV: IVONE CONCEIÇÃO SILVA (OAB 13609B/MS) ADV: ÉZIO PEDRO FULAN (OAB 12173A/MS) ADV: MATILDE DUARTE GONÇALVES (OAB 12174A/MS) ADV: CLEVERSSON GOLIN '
    '(OAB 14452/MS) Ante o resultado negativo do leilão eletrônico, fica a exequente intimada para requerer o que de direito no prazo de 15 (quinze) dias.')),

    build_deadline_case_test(('Processo 0810454-03.2017.8.12.0001 - Recuperação Judicial - Recuperação judicial e Falência Reqte: SF - Fôrmas para Construção Civil Ltda e '
    'outros - Administra: VCP - Vinicius Coutinho Consultoria e Perícia S/S Ltda ADV: KARYNA HIRANO DOS SANTOS (OAB 9999/MS) ADV: EDUARDO HENRIQUE VIEIRA BARROS '
    '(OAB 7680O/MT) ADV: BERNARDO RODRIGUES DE OLIVEIRA CASTRO (OAB 14992A/MT) ADV: FLÁVIO NERY COUTINHO SANTOS CRUZ (OAB 51879/MG) ADV: EDUARDO HENRIQUE VIEIRA BARROS '
    '(OAB 21019A/MS) ADV: BÁRBARA BRUNETTO (OAB 20128/MT) ADV: EUCLIDES RIBEIRO SILVA JUNIOR (OAB 21020A/MS) ADV: FELIPE PALHARES GUERRA LAGES (OAB 84632/MG) ADV: '
    'MAYRAN OLIVEIRA DE AGUIAR (OAB 122910/MG) Trata-se de pedido, formulado pelas Recuperandas, a fim de que sejam liberados os valores retidos pelo Banco Safra S. A., '
    'bem como para que este se abstenha de promover novos débitos, especialmente os referentes às duplicatas mercantis futuras, e seja determinado à '
    'Energisa Mato Grosso do Sul - Distribuidora de Energia S. A, a devolução do valor pago, objeto de decisão liminar (fls. 582-587). Pois bem. '
    'É sabido que o sistema de \'trava bancária\' é um instrumento contratual empregado por instituições financeiras em operações de crédito, como forma de garantia da '
    'dívida decorrente de financiamento. Conforme esse sistema, a instituição financeira, em caso de inadimplência do devedor, pode receber seus créditos a partir dos '
    'depósitos recebíveis pelo financiado, os quais são depositados em conta-corrente vinculada, controlada pela instituição financeira, credora, e de titularidade do '
    'devedor. No caso dos autos, a instituição financeira Banco Safra S. A., está se utilizando do referido sistema e retendo os valores referentes aos recebíveis das '
    'Recuperandas. Registre-se que os pedidos das Recuperandas estão lastreados pelos princípios preservadores da empresa. Com efeito, a filosofia que norteia a própria '
    'Lei de Recuperação Judicial (Lei n. 11.101/05), é a que preserva a empresa como cédula da sociedade. Assim preceitua o art. 47 da Lei n. 11.101/05, '
    'in verbis:\'Art. 47. A recuperação judicial tem por objetivo viabilizar a superação da situação de crise econômico-financeira do devedor, a fim de permitir a '
    'manutenção da fonte produtora, do emprego dos trabalhadores e dos interesses dos credores, promovendo, assim, a preservação da empresa, sua função social e o estímulo '
    'à atividade econômica\'. No presente caso, com o fim de preservar o interesse de todos os credores, viabilizar a continuação empresarial, a manutenção de empregos, '
    'a liquidez do capital de giro, entendo que deve ser determinada a suspensão da trava bancária a fim de possibilitar a recuperação das empresas. Nota-se que a '
    'preservação da empresa é de interesse público. Não se pode, portanto, estabelecer privilégios para qualquer classe de credores. Os Tribunais de Justiça, em '
    'julgamentos recentes, admitem a liberação da trava bancária em sede de recuperação judicial, com vistas a possibilitar o sucesso da recuperação e a preservação da '
    'empresa. O nosso Egrégio Tribunal de Justiça de Mato Grosso do Sul, em inúmeros Acórdãos manteve as decisões de primeiro grau, determinando a liberação das travas '
    'bancárias, com base no principio da preservação da empresa, senão vejamos:E M E N T A - AGRAVO DE INSTRUMENTO - RECUPERAÇÃO JUDICIAL - PRELIMINAR DE OFENSA AO '
    'PRINCÍPIO DA DIALETICIDADE REJEITADA - PRELIMINAR DE PRECLUSÃO - CONFUNDE SE COM O MÉRITO - DESCONSTITUIÇÃO DAS TRAVAS BANCÁRIAS PELO JULGADOR SINGULAR - '
    'NECESSIDADE DE MANUTENÇÃO DA ORDEM DE LIBERAÇÃO DO ACESSO DA EMPRESA RECUPERANDA A CONTAS E AOS VALORES QUE NELA CONSTAM - DESNECESSIDADE DO REGISTRO DO CONTRATO '
    'JUNTO AO CARTÓRIO DE TÍTULOS E DOCUMENTOS PARA A VALIDADE DA CONDIÇÃO PRIVILEGIADA DO CREDOR FIDUCIÁRIO - NECESSIDADE DE PRESERVAÇÃO DA EMPRESA EM RECUPERAÇÃO - '
    'RECURSO CONHECIDO E DESPROVIDO. A C Ó R D Ã OVistos, relatados e discutidos estes autos, acordam os juízes da 5ª Câmara Cível do Tribunal de Justiça, na '
    'conformidade da ata de julgamentos, por unanimidade, afastar as preliminares, nos termos do voto do Relator e, no mérito, por maioria, negar provimento ao recurso, '
    'nos termos do voto do 1º Vogal, vencido o Relator. Campo Grande, 18 de abril de 2017. Des. Júlio Roberto Siqueira Cardoso - Relator designadoD E C I S Ã OComo consta '
    'na ata, a decisão foi a seguinte:POR UNANIMIDADE, AFASTARAM AS PRELIMINARES, NOS TERMOS DO VOTO DO RELATOR E, NO MÉRITO, POR MAIORIA, NEGARAM PROVIMENTO AO RECURSO, '
    'NOS TERMOS DO VOTO DO 1º VOGAL, VENCIDO O RELATOR. Presidência do Exmo. Sr. Des. Luiz Tadeu Barbosa Silva Relator, o Exmo. Sr. Des. Luiz Tadeu Barbosa Silva. Tomaram '
    'parte no julgamento os Exmos. Srs. Des. Luiz Tadeu Barbosa Silva, Des. Júlio Roberto Siqueira Cardoso e Des. Sideni Soncini Pimentel. Campo Grande, 18 de abril de '
    '2017. Neste sentido, já decidiu o Tribunal de Justiça do Rio de Janeiro:0038015-14.2015.8.19.0000 - AGRAVO DE INSTRUMENTO 1ª Ementa DES. SERGIO NOGUEIRA DE '
    'AZEREDO - Julgamento: 05/04/2016 - DECIMA NONA CAMARA CÍVEL Agravo de Instrumento. Recuperação Judicial. Antecipação dos efeitos da tutela deferida para a parcial '
    'liberação de \'trava bancária\', decorrente de mútuo avençado com cessão fiduciária de direitos creditórios. Irresignação. Possibilidade de solução monocrática. '
    'Verbete nº 59 da Súmula da Jurisprudência Predominante deste Egrégio Tribunal de Justiça. Decisum que apresenta a necessária fundamentação, amparada em '
    'interpretação da legislação pertinente conferida pelos Tribunais pátrios e nas provas dos autos, e que não se revela extravagante diante dos contornos da matéria '
    'apreciada. Procedimento recuperatório respaldado nos Princípios da Preservação da Empresa e da sua Função Social. Finalidade precípua que consiste em viabilizar '
    'o soerguimento e reestruturação da Demandante, com o intuito de preservar o interesse daqueles atingidos por sua debilidade financeira trabalhadores, credores e '
    'sociedade -, por meio de concessão de lapso temporal para a elaboração de plano de saneamento, bem como de medidas legais inibitórias da redução do seu patrimônio '
    'por atos de constrição no período. Controvérsia acerca da natureza dos créditos que não afasta a vedação legal ao esvaziamento da empresa recuperanda quanto aos bens '
    'necessários à manutenção de sua atividade econômica. Inteligência da norma limitadora de direitos prevista no art. 49, §3º, da Lei nº 11.101/05. Regra de exceção, '
    'cuja boa hermenêutica impõe interpretação restritiva, vedada qualquer forma de presunção, analogia ou ampliação. Imprescindibilidade do destravamento dos valores '
    'retidos ao cumprimento do programa recuperatório. Imperativa ponderação dos interesses conflitantes que conduz à necessária prevalência, por ora, do objetivo '
    'prioritário da legislação regente sobre a pretensão da Agravante quanto à imediata satisfação de seu crédito. Precedentes desta Colenda Corte. Manutenção da '
    'decisão agravada que se impõe. Desprovimento do recurso, com fulcro no art. 932, IV, \'a\' do CPC. Assim como o Tribunal de Justiça do Estado de Minas Gerais:Agravo '
    'de Instrumento-Cv 1.0024.12.273566-5/0011189115-69.2012.8.13.0000 Relator(a): Des.(a) Judimar Biber Data de Julgamento: 13/06/2013 Data da publicação da súmula: '
    '21/06/2013 O princípio da preservação da empresa, insculpido no art. 47 da Lei Federal 11.101/05, dispõe que a recuperação judicial tem por objetivo viabilizar '
    'a superação da situação de crise econômico-financeira do devedor, a fim de permitir a manutenção da fonte produtora, do emprego dos trabalhadores e dos '
    'interesses dos credores, promovendo, assim, a preservação daquela, sua função social e o estímulo à atividade econômica, condição que tudo indica que não será '
    'alcançada acaso não se inclua os créditos fiduciários não sujeitos à recuperação e que levará a inevitável falência da empresa, de modo que o afastamento da '
    'trava bancária se mostraria procedimento de natureza cautelar que preserva as condições preferenciais dos créditos na iminente falência que já é conhecida acaso não '
    'se abra mão da garantia ofertada em prol do interesse público de propiciar a recuperação. Não provido. Nota-se que os E. Tribunais de todo o pais estão mantendo as '
    'decisões proferidas em primeiro grau, liberando a \'trava bancária\', mesmo não sendo os créditos sujeitos ao processo de recuperação judicial, visto que os '
    'princípios previstos no art. 47 da Lei n. 11.101/05, que regem o processo referido, têm por objetivo viabilizar a superação da situação de crise '
    'econômico-financeira do devedor, a fim de permitir a manutenção da fonte produtora, do emprego dos trabalhadores e dos interesses dos credores. E para que isso '
    'aconteça, é necessário que se disponibilize ao empresário devedor os valores depositados em suas contas bancárias, visando também à manutenção do capital de giro. '
    'Só assim, a ordem determinada pelo artigo legal referido poderá ter eficácia, possibilitando o soerguimento da empresa. Adoto a motivação dos Acórdãos '
    'supramencionados como fundamentação da presente decisão, que interpretaram adequadamente a legislação de falência e de recuperação de empresas, adotando o '
    'princípio maior da referida lei, qual seja, o da preservação da empresa. Se não ocorrer a liberação das \'travas bancárias\', para que as empresas mantenham o '
    'seu capital de giro, vislumbra-se a real possibilidade da declaração da falência, situação que contribui para o caos social e aumento da crise econômica nesta '
    'capital. Ressalta-se que este juízo já declarou na decisão inicial a inconstitucionalidade dos §§ 3º e 4º do art. 49 da Lei n. 11.101/2005, determinando a sujeição '
    'dos créditos bancários ao processo de recuperação da empresa, portanto, não há razão alguma, para se manter as \'travas bancárias\', pois, no caso em tela, deve, '
    'com certeza, prevalecer as normas constitucionais previstas no art. 172 da Constituição Federal de 1988. Do exposto:1. Determino a suspensão das travas '
    'bancárias referentes aos contratos firmados entre as Recuperandas e a instituição financeira. No entanto, deixo por ora de analisar a primeira '
    'parte do item \'a\' de fl. 641, ou seja, acerca da devolução do valor mencionado pelas Recuperandas, a saber, R$558.685,68 (quinhentos e cinquenta e oito mil, '
    'seiscentos e oitenta e cinco reais e sessenta e oito centavos), eis que dependente de verificação pormenorizada por este Juízo. Ademais, intime-se o AJ para que '
    'se manifeste, no prazo de 48 (quarenta e oito) horas, a respeito deste valor indicado pelas Recuperandas. Após a manifestação, voltem os autos conclusos para '
    'análise.2. No tocante ao pedido de abstenção, última parte do item \'a\' de fl. 641, defiro o pleito para que o Banco Safra S. A. se abstenha de efetuar novos '
    'débitos, especialmente aqueles referentes às duplicatas mercantis futuras nas contas relacionadas à fl. 641, relativa a tais obrigações, uma vez demonstrada a '
    'essencialidade dos valores retidos pelo referido banco à atividade empresarial do Grupo SF Fôrmas, sob pena de penhora na boca do caixa. Para tanto, oficie-se '
    'com urgência à referida instituição bancária para que cumpram a presente decisão, no prazo de 5 (cinco) dias.3. Quanto ao pedido relacionado à devolução do valor '
    'pago à empresa energética, defiro o item \'b\' de fls. 641-642 e determino a expedição de ofício à Energisa Mato Grosso do Sul - Distribuidora de Energia S. A., '
    'para que restitua o valor pago de R$12.083,20 (doze mil, oitenta e três reais e vinte centavos), para a conta-corrente de titularidade da SF Fôrmas (Agência n. '
    '2140, conta-corrente n. 13003240-4, banco Santander), bem como cumpra o determinado às fls. 582-587, realizando o desmembramento da fatura no valor de R$1.102,51 '
    '(um mil,cento e dois reais e cinquenta e um centavos).4. Ciente da juntada, realizada pelas Recuperandas, dos documentos de fls. 643-731/1.020-1.024.5. '
    'Quanto à manifestação de fls. 740-747, deve a ora habilitante proceder na forma do § 1º do art. 7º da Lei n. 11.101/05, conforme determinado na decisão que '
    'deferiu o processamento da presente recuperação (fls. 485-538), ou seja, encaminhar a divergência apontada ao AJ nomeado nos autos.6. Comunique-se o AJ acerca do '
    'teor da petição de fls. 748- 752, formulada pela Fazenda Pública do Município de Campo Grande-MS.7. Cadastrem-se nos autos os advogados indicados às fls. '
    '460, 728, 779, 784-787 e 1.056.8. Ciente da comunicação dos pedidos de habilitação dos seguintes credores: Banco do Brasil S/A (fls. 781-1.019); Nicola Cardinale '
    'e outros (fls. 1.032-1.055); e Banco Safra S. A. (fls. 1.056-1.074), todos encaminhados ao AJ, bem como do Ofício n. 265/2017 da JUCEMS (fls. 1.025-1.030). 9. '
    'Por fim, defiro o pedido de fl. 1.032 e concedo o prazo de 5 (cinco) dias para regularização da representação processual.')),

    build_deadline_case_test(('31 48-53.2014.4.01.9360 MANDADO DE SEGURANCA CIVEL/TR Juiz Relator: FÁBIO HENRIQUE RODRIGUES DE MORAES FIORENZA IMPTE:ERASMO SOARES DELGADO '
    'ADVOGADO:MT00018378 - ADRIANE SANTOS DOS ANJOS ADVOGADO:MT00006658 - JOAO BATISTA DOS ANJOS ADVOGADO:MT0010327B - ADILIO HENRIQUE DA COSTA LITISPA:INSTITUTO NACIONAL '
    'DO SEGURO SOCIAL-INSS IMPDO:JUIZ DO JUIZADO ESPECIAL FEDERAL (6 VARA FEDERAL) ATO ORDINATÓRIO: Intimação da parte autora, para que apresente contrarrazões ao Incidente '
    'de Uniformização à TNU interposto pela parte Ré (Prazo: quinze dias).')),

    build_deadline_case_test(('INTIMAÇÃO EFETIVADA REF. À MOV. CONHECIDO E NÃO-PROVIDO - 07/07/2017 10:33:10 LOCAL : 2ª TURMA JULGADORA MISTA DA 2ª REGIÃO '
    '(APARECIDA DE GOIÂNIA) NR. PROCESSO : 5111195.18.2015.8.09.0012 CLASSE PROCESSUAL : Procedimento do Juizado Especial Cível POLO ATIVO : VINICIUS ALVES DE '
    'ALMEIDA POLO PASSIVO : TIM CELULAR S/A SEGREDO JUSTIÇA : NÃO PARTE INTIMADA : TIM CELULAR S/A ADVG. PARTE : 34847 GO - FELIPE GAZOLA VIEIRA MARQUES PARTE INTIMADA : '
    'VINICIUS ALVES DE ALMEIDA ADVG. PARTE : 33761 GO - RAFAEL RODRIGUES CAETANO - VIDE ABAIXO O(S) ARQUIVO(S) DA INTIMAÇÃO. ESTADO DE GOIÁS PODER JUDICIARIO COMARCA DE '
    'APARECIDA DE GOIÂNIA 2ª Turma Julgadora Mista da 2ª Região (Aparecida de Goiânia) Av. Fórum, s/n, CENTRO, APARECIDA DE GOIÂNIA - Fone: (62) Ação: Procedimento do '
    'Juizado Especial Cível Processo nº: 5111195.18.2015.8.09.0012 Promovente(s): VINICIUS ALVES DE ALMEIDA Promovido(s): TIM CELULAR S/A DECISÃO MONOCRÁTICA Apesar de '
    'dispensável, procedo a breve relatório. Trata-se de RECURSO INOMINADO apresentado por TIM CELULAR S/A contra sentença que julgou procedente os pedidos formulados '
    'por VINÍCIUS ALVES DE ALMEIDA. De início, o feito cuida de Ação Declaratória de Inexistência de Débitos c/c Indenização por Danos Morais ajuizada pelo recorrido '
    'em face da recorrente. O processo teve curso normal, vindo a ser proferido sentença no evento 12, que julgou procedentes os pedidos formulados pelo autor. A '
    'parte demandada apresentou Recurso Inominado, pugnando pela reforma da sentença. Em suma, afirma que a inscrição do autor nos cadastros negativos de crédito foi '
    'legal, pois o débito é legítimo. Afirma, ainda, que o fato não imprimiu dano moral ao autor, mas meros aborrecimentos de rotina, pugnando, em caso de rejeição '
    'das teses, que o valor da indenização seja reduzido. A parte recorrida apresentou suas Contrarrazões (ev. 23), requerendo o improvimento do recurso. '
    'Distribuído ao relator, este converteu o feito em diligência e determinou que a parte demandada juntasse a íntegra das ocorrências de inscrição em nome do autor '
    'nos órgão de proteção ao crédito (ev. 40). O autor juntou complementação ao documento no evento 45, mas não foi aceito pelo e. Relator, sendo reaberta a '
    'oportunidade. A requerida/recorrente, apresentou documentos no ev. 51, e o autor/recorrido, no ev. 58. É o relatório. Decido. NR. PROCESSO: 5111195.18.2015.8.09.0012 '
    'Inicialmente, observo que a causa sob análise permite o julgamento monocrático na forma do art. 932, do Código de Processo Civil, afinal, a matéria está '
    'amplamente discutida e a questão é meramente de direito, existindo ampla jurisprudência firmada nos Tribunais e Turmas Recursais sobre a matéria de cerne sob '
    'análise, como se verá. Assim, passo a decidir o mérito recursal monocraticamente. Juízo de admissibilidade Compulsando os autos com atenção, é possível '
    'observar que o recurso é tempestivo e o preparo foi recolhido no prazo e forma adequados, motivo pelo qual, conheço do recurso. Do Mérito Trata-se de recurso '
    'inominado interposto pela TIM S/A contra a r. sentença proferida no evento 12, que julgou parcialmente procedente os pedidos formulados na ação inicial na '
    'ação movida pelo autor. Primeiramente, cabe salientar que as provas novas juntadas pelas partes em grau de recurso não devem ser consideradas, especialmente se '
    'se tratam de provas que já estavam a disposição das partes na época do ajuizamento da ação. Ocorre, nesses casos, preclusão temporal e consumativa, pois tratam de '
    'fatos que ocorreram antes do ajuizamento da ação e que estavam disponíveis para serem objeto de provação nos autos no momento oportuno. Ademais, saliento que '
    'o artigo 33 da Lei nº 9.099/95 determina que todas as provas sejam produzidas no momento da audiência de instrução, tendo as partes declarado, desde a '
    'audiência de conciliação, a inexistência de outras provas a serem produzidas, conforme se verifica do evento 10. De tal sorte, não podem ser conhecidos os '
    'documentos colacionados às razões recursais, porquanto extemporâneos, na medida em que a oportunidade instrutória foi atingida pela preclusão. Assim, resta prejudicada '
    'a análise dos documentos - telas de sistemas de informática - ainda que, nesse caso, tenham sido objeto de solicitação do e. Relator do recurso na época. '
    'Cumpre destacar que a matéria suscitada não é de ordem pública, mas de prova, e cada parte tem seu ônus probatório devidamente distribuído pela regra do art. 373, '
    'do CPC, combinado, nesse caso, com as regras dos arts. 6º, inc. VIII, do CDC. Nesse sentido, apenas para corroborar, trago trecho de julgado da Turma: \'(...) A '
    'juntada de provas com as razões recursais não é admissível, em regra, no sistema processual civil brasileiro, o que só é possível na hipótese de documento novo, '
    'que não poderia ser apresentado por extrema impossibilidade, situação esta que não se verificou no caso em exame, assim tais documentos não serão agora analisados, '
    'pois avulta, forçosamente, no fenômeno da preclusão. (…)\' (TJGO, RECURSO CIVEL 2009046949110000, Rel. DR(A). LUIS ANTONIO ALVES BEZERRA, TURMA JULGADORA RECURSAL CIVEL '
    'DOS JUIZADOS ESPECIAIS, julgado em 21/06/2010, DJe 616 de 09/07/2010). Quanto à tese recursal, observo que, como dito, o ônus de provar a regularidade da dívida e, '
    'portanto, da inscrição feita perante os órgãos de proteção ao consumo é da parte requerida, não tendo esta, entretanto, suprido seu ônus probatório, pois deixou de '
    'provar de forma coerente e indúbia que, de fato, existia relação comercial estabelecida entre ela e o autor. NR. PROCESSO: 5111195.18.2015.8.09.0012 Não trouxe a '
    'requerida/recorrente, nenhum comprovante legítimo de que o autor, de fato, teria contratado seus serviços. Na verdade, não trouxe nenhum documento nesse sentido. '
    'Assim, inviável o acolhimento de sua tese. Quanto aos danos morais, observa-se que os fatos narrados transbordam a mera adversidade ou frustrações do dia-a-dia, '
    'configurando dano moral a conduta temerária e desrespeitosa da operadora telefônica que, apesar da insistência do autor, continua a proceder a manutenção de seu nome '
    'negativado. Por sua vez, sobre o \'quantum\' indenizatório dos danos morais, observo que deve ser fixado em consonância com o princípio da razoabilidade, não podendo, '
    'de forma alguma, ser causador de enriquecimento ilícito do ofendido nem ser irrisório a ponto de não ter nenhum efeito reparatório ou pedagógico. A propósito, o que '
    'vem prevalecendo no C. STJ: \'O critério que vem sendo utilizado por esta Corte Superior na fixação do valor da indenização por danos morais, considera as condições '
    'pessoais e econômicas das partes, devendo o arbitramento operar-se com moderação e razoabilidade, atento à realidade de vida e às peculiaridades de cada caso, de forma '
    'a não haver o enriquecimento indevido do ofendido, bem como que sirva para desestimular o ofensor a repetir o ato ilícito...\' (Resp. n. 913.131-BA. Rel. Min. Carlos '
    'Fernando Mathias. j. 16.9.2008). No caso \'sub judice\', atendidas as peculiaridades do caso concreto e levando-se em consideração os inconvenientes suportados pelo '
    'autor, não há que se falar que o \'quantum\' indenizatório fixado no \'decisum\' (R$ 10.000,00) é excessivo. Tal quantia está em consonância com a jurisprudência '
    'e é suficiente para desestimular a ofensora a repetir o ato. Nesse sentido a jurisprudência das Turmas Recursais em caso '
    'semelhante: \'RECURSO INOMINADO. AÇÃO DECLARATÓRIA DE INEXISTÊNCIA DE DÉBITO COM DANOS MORAIS. TELEFONIA. ALTERAÇÃO CONTRATUAL EM DECORRÊNCIA DA MÁ PRESTAÇÃO DE '
    'SERVIÇO. COBRANÇA INDEVIDA DE MULTA RESCISÓRIA NEGATIVAÇÃO INDEVIDA. DANO MORAL CONFIGURADO. SENTENÇA REFORMADA. RECURSO CONHECIDO E IMPROVIDO. I - Omissis. II - '
    'Omissis. III - Deve a empresa de telefonia indenizar, por se tratar de dano moral in re ipsa, conforme inteligência do art. 14 do CDC. A recorrida somente '
    'estaria isenta da responsabilidade se demonstrasse culpa exclusiva do consumidor ou de terceiro (artigo 14, § 3º, inciso II, do CDC), o que não ocorreu. IV - A '
    'reparação do dano moral deve servir para recompor os transtornos sofridos pela vítima, bem como para inibir a repetição de ações lesivas da mesma natureza, '
    'motivo pelo qual a sua fixação deve ser moderada, sem perder o seu caráter pedagógico e punitivo. recurso conhecido e provido para declarar inexistente o '
    'débito oriundo do contrato nº 207380738. danos morais arbitrados em R$ 10.000,00 (dez mil reais) com incidência de correção monetária pelo índice inpc, a partir '
    'da fixação, mais juros de 1%% ao mês retroagidos a data da citação, por se tratar de relação contratual. Súmula 362 do STJ e artigo 405 do Código Civil. Sem '
    'custas e honorários nos termos do artigo 55 da lei 9.099/95. (TJGO, RECURSO CIVEL 2011940853500000, Rel. DR(A). TARSIO RICARDO DE OLIVEIRA FREITAS, TURMA JULGADORA '
    'RECURSAL CIVEL DOS JUIZADOS ESPECIAIS, julgado em 31/07/2015, DJe 1845 de NR. PROCESSO: 5111195.18.2015.8.09.0012 11/08/2015)\' (grifei). Pelo exposto, conheço '
    'do recurso, mas lhe NEGO PROVIMENTO, mantendo a sentença em seus termos. Custas e honorários pela recorrente, que fixo em 20%% sobre o valor da condenação. '
    'Intimem. Transitada em julgado, deem baixa de minha relatoria no sistema de 2º grau dos Juizados Especiais Cíveis e Criminais. Retirem de pauta de julgamento. '
    'APARECIDA DE GOIÂNIA, em 7 de julho de 2017 Aluízio Martins Pereira de Souza Juiz de Direito - Relator NR. PROCESSO: 5111195.18.2015.8.09.0012')),

    build_deadline_case_test(('Processo 0800344-17.2016.8.12.0053 - Procedimento Comum - Empréstimo consignado Autora: Naurelina Reginaldo da Silva - Réu: Banco '
    'Votorantim ADV: ANDRE LUIZ BOLDRIN CARDOSO (OAB 18743/MS) ADV: RODRIGO SCOPEL (OAB 18640A/MS) ADV: JULIANO FRANCISCO DA ROSA (OAB 18601A/MS) ADV: LUIZ '
    'FERNANDO CARDOSO RAMOS (OAB 14572/MS) Passo ao saneamento, nos termos do art. 357 do CPC. Não há questões processuais pendentes. As partes são capazes e estão '
    'devidamente representadas nos autos. A matéria não é complexa quanto ao fato ou direito. Passo à análise das preliminares arguidas pelo requerido. A) Da retificação '
    'do polo passivoPreliminarmente, o banco réu pugna pela retificação do polo passivo da demanda, para que conste sua verdadeira razão social, qual seja BV '
    'FINANCEIRA - CRÉDITO, FINANCIAMENTO E INVESTIMENTO, instituição responsável pela operacionalização do produto objeto da demanda. Pois bem. A partir das razões '
    'ofertadas, bem como do \'convênio para cessão de direitos e obrigações de crédito consignado - INSS\' (fls. 89-93), não vejo prejuízo em acolher tal pedido, tendo em '
    'vista que BV FINANCEIRA S. A - CRÉDITO, FINANCIAMENTO E INVESTIMENTO, inscrita no CNPJ n.01.149- 953/0001-89, apesar de possuir estrutura e atividade própria, é do '
    'mesmo conglomerado do Banco Votorantim S. A. Assim, ACOLHO o pedido de retificação do polo passivo. Anote-se. B) Da prescriçãoNeste ponto impende esclarecer que nos '
    'casos de responsabilidade pelo fato do produto e do serviço, aplica-se o prazo prescricional de 05 anos, nos termos do disposto no art. 27 do Código de Defesa do '
    'Consumidor, in verbis:\'Art. 27. Prescreve em cinco anos a pretensão à reparação pelos danos causados por fato do produto ou do serviço prevista na Seção II deste '
    'Capítulo, iniciando-se a contagem do prazo a partir do conhecimento do dano e de sua autoria\'. E, no presente caso, trata-se de típica relação de consumo, o que '
    'atrai a incidência do CDC. Em razão disso, REJEITO a preliminar de prescrição trienal arguida. Por outro lado, tendo em vista que se trata de declaratória de '
    'inexistência de débito em decorrência de um contrato de empréstimo supostamente realizado entre requerente e réu, é certo que se configura em uma obrigação de '
    'trato sucessivo, porquanto diz respeito a descontos de parcelas mensais, cuja violação do direito ocorre de forma contínua. Destarte, o prazo da prescrição corre a '
    'partir de cada desconto da parcela prevista no contrato, porque o dano e sua autoria se tornaram conhecidos com cada débito no benefício previdenciário do suplicante. '
    'Nesse sentido, trago à baila recente julgado do e. TJMS sobre o tema: APELAÇÃO CÍVEL - AÇÃO DECLARATÓRIA DE INEXISTÊNCIA DE DÉBITO C/C RESTITUIÇÃO DE VALORES E DANOS '
    'MORAIS - EMPRÉSTIMOS CONSIGNADOS - BENEFÍCIO PREVIDENCIÁRIO - PRESCRIÇÃO QUINQUENAL - A PARTIR DO ÚLTIMO DESCONTO - APLICAÇÃO DO ARTIGO 27 DO CDC - '
    'PARCIAL ACOLHIMENTO - ASSINATURA A ROGO SEM AS FORMALIDADES LEGAIS - PESSOA ANALFABETA - CONTRATO INVÁLIDO - FALHA NA PRESTAÇÃO DO SERVIÇO - RESPONSABILIDADE CIVIL '
    'VERIFICADA - DANOS MORAIS CONFIGURADOS - QUANTUM INDENIZATÓRIO - RESTITUIÇÃO DO INDÉBITO - FALTA DE INTERESSE RECURSAL - INCIDÊNCIA DE JUROS - MASSA FALIDA EM '
    'PROCESSO DE LIQUIDAÇÃO - POSSIBILIDADE - RECURSO PARCIALMENTE CONHECIDO E, NESTA PARTE, PARCIALMENTE PROVIDO. 1. O prazo prescricional da pretensão de reparação '
    'decorrente de empréstimo consignado ilícito com descontos mensais é de cinco anos, nos termos do art. 27 do CDC, contados da última prestação indevidamente subtraída. '
    '(...). (Apelação - Nº 0801842-88.2014.8.12.0031- Relator(a): Juiz Jairo Roberto de Quadros; Comarca: Caarapó; Órgão julgador: 2ª Câmara Cível; Data do julgamento: '
    '20/07/2016; Data de registro: 29/07/2016). (sem grifo no original)Consoante análise do conjunto probatório acostado aos autos, tenho que o feito foi ajuizado em outubro '
    'de 2016, ou seja, mais de 05 (cinco) anos após o término dos descontos do contrato nº 191310318, haja vista que os descontos se iniciaram no mês de agosto de 2007, e se '
    'encerraram em setembro de 2009, sendo imperioso reconhecer a prescrição da pretensão autoral no que se refere ao contrato nº 191310318. No mais, em relação ao contrato '
    'nº 194278550, consigno que a prescrição alcança a pretensão autoral apenas no que tange às parcelas descontadas há mais de 05 (cinco) anos da data do ajuizamento '
    'da demanda (outubro/2016), ou seja, outubro/2011. Ultrapassadas as preliminares, quanto aos fatos controversos, delimito a questão sobre a qual recairá a '
    'atividade probatória como sendo: a) se houve ou não a contratação dos serviços que deram origem ao débito, bem como sua efetiva contraprestação por parte do '
    'réu, qual seja, o recebimento dos valores pela parte autora; b) a extensão dos danos decorrentes dos supostos descontos indevidos no benefício previdenciário '
    'da parte autora. O ônus da prova recairá sobre o réu, visto que aplicável, no presente caso, o Código de Defesa do Consumidor, mais especificamente, o art. 6º, VIII, '
    'para fins de garantir a facilitação da defesa do direito ao consumidor. Isso porque este Juízo reputa verossímil a alegação da parte autora, bem como reconheço sua '
    'condição de hipossuficiente, notadamente pelo fato de litigar em desfavor de Instituição Bancária, a qual possuiria eventual documentação passível de '
    'descaracterização da pretensão inaugural. Quanto ao pedido de expedição de ofício, verifico que merece ser parcialmente deferido. Consigno que tal '
    'entendimento não está em desacordo com a inversão do ônus da prova, a qual protege os consumidores, visto que ainda cabe ao fornecedor produzir as provas '
    'para ilidir a pretensão autoral, o que se constata através do pedido em questão. Ademais, é de conhecimento deste Juízo o elevado número de ações dessa natureza '
    'ajuizadas no Estado de Mato Grosso do Sul, em especial, nesta Comarca, sendo que a expedição de ofício revela-se necessária para uma melhor análise de cada '
    'caso em concreto. Por fim, o pedido somente poderá ser deferido quando plausíveis as alegações e documentos apresentados em contestação, o que, de fato, foi '
    'demonstrado às fls. 99-108, visto que o requerido juntou, inclusive, cópia de contrato supostamente celebrado pela parte autora. Portanto, ao menos em sede de '
    'cognição sumária, visto que não está sendo feito, por ora, o julgamento de mérito da causa, o pleito comporta acolhimento. Todavia, tendo em vista que o '
    'contrato a que se refere o comprovante de saque solicitado foi declarado prescrito, não há necessidade de atendimento ao quanto requerido no item \'c\' da petição '
    'de fl. 137. Quanto ao pedido de produção de prova pericial, defiro-o apenas em relação ao contrato nº 194278550, porquanto entendo necessária ao julgamento do '
    'mérito, a fim de verificar a autenticidade da assinatura/ digital nos documentos juntados pelo requerido. Assim, visto que expressamente requeridas pela parte, bem '
    'como necessárias para resolução da lide, admito, a título de prova, a juntada de novos documentos (resposta ao ofício), perícia e o depoimento pessoal da parte '
    'autora. Ante o exposto:1. Nos termos do art. 356, II c/c art. 487, II, ambos do CPC, resolvo, parcialmente, o mérito, para declarar a prescrição da pretensão '
    'autoral em relação ao contrato nº 191310318, bem como em relação às parcelas do contrato nº 194278550 descontadas antes de outubro/2011. Condeno a parte autora ao '
    'pagamento de 50% (cinquenta por cento) das despesas e custas processuais, bem como honorários advocatícios, os quais fixo no valor de R$ 500,00 (quinhentos reais), '
    'com fulcro no art. 85, § 8º, do CPC, ante o baixo valor dos contratos, devendo ser suspensa sua exigibilidade, visto que deferida a gratuidade da justiça à '
    'parte autora (fls. 57-58), conforme art. 98, § 1º, do CPC. P. R. I. C.2. Sem prejuízo, expeça-se ofício, nos exatos termos requeridos no item \'d\' da petição de fl. '
    '137. Com a resposta, deverá, a Serventia, providenciar o sigilo do documento juntado aos autos, o qual deverá ser disponibilizado apenas às partes e a este Juízo. '
    'Após, dê-se vista às partes, para manifestação no prazo de 05 (cinco) dias.3. Para realização da perícia, '
    'nomeio a empresa \'Vinicíus Coutinho Consultoria e Perícia - VCP\', representada pelo seu diretor, Sr. Vinícius Coutinho, com sede à Rua da Treze de Maio, n. 2500, '
    'sala 1307, 13º andar, Centro, PABX: 3389-3000, Campo Grande, MS, e-mail: vcp@vcpericia.com.br, devendo ela ser intimada da presente nomeação a fim de '
    'apresentar sua proposta de honorários em 5 (cinco) dias. Apresentada a proposta, abra-se vista às partes para manifestação, no prazo de 5 (cinco) dias. '
    'Em atenção ao art. 95, caput, do CPC, os honorários deverão ser arcados pelo requerido. Assim, havendo concordância com o valor dos honorários, intime-se para '
    'depositar em juízo o valor correspondente. Intime-se, ainda, as partes para que, dentro de 15 (quinze) dias desta decisão, apresentem eventuais quesitos, '
    'indiquem assistente técnico ou manifestem impedimento ou suspeição do perito, se for o caso. Como quesito do Juízo, logo indico: a) identificar se as '
    'digitais/assinaturas constantes dos documentos pertinentes à contratação questionada pertencem à parte autora. Cumpridas as determinações acima, deverá o '
    'expert fixar dia e hora para o início dos trabalhos, intimando-se as partes e seus assistentes da data e horário estabelecidos. O laudo pericial deverá vir aos '
    'autos no prazo de 30 (trinta) dias, contados da data do início da perícia. Defiro o pagamento de até 50% (cinquenta por cento) dos honorários arbitrados a '
    'favor do perito no início dos trabalhos, devendo o remanescente ser pago apenas ao final, depois de entregue o laudo e prestados todos os esclarecimentos '
    'necessários. Com a juntada do laudo pericial nos autos, intimem-se as partes para manifestação.4. Desde já, designo audiência de instrução e julgamento para o '
    'dia 9.8.2017, às 13h15, devendo ser intimada a parte autora para depoimento pessoal, sob pena de confissão (art. 385, §1º, CPC).5. Por fim, defiro o prazo de 15 '
    '(quinze) dias para que o requerido promova a entrega, no cartório deste Juízo, dos documentos originais que serão periciados (contrato nº 194278550).6. Dê-se ciência, '
    'à parte autora, acerca do documento juntado aos autos à fl. 139.7. As partes têm o direito de pedir esclarecimentos ou solicitar ajustes, no prazo comum de 5 '
    '(cinco) dias, findo o qual a presente decisão se tornará estável (art. 357, §1º, CPC).')),

    build_deadline_case_test(('NR. PROTOCOLO : 324129-30.2015.8.09.0006 AUTOS NR. : 828 NATUREZA : RESCISAO CONTRATUAL REQUERENTE : CLAUDIO EDUARDO SALEM ELIAS ETELVINA '
    'FILOMENA CORREIA MARQUES ELIAS REQUERIDO : TERRAS ALPHA ANAPOLIS EMPREENDIMENTOS IMOBILIARI VIA ANAPOLIS LTDA ADV REQTE : 35611 GO - JONEY VILELA ANDRADE '
    'JUNIOR 36222 GO - ODILON PONCIANO DIAS NETTO ADV REQDO : 169451 SP - LUCIANA NAZIMA 213416 SP - GISELE CASAL KAKAZU DESPACHO : PROCESSO N: 201503241291 SENTENCA '
    'CUIDA-SE DE ACAO DE RESCISAO CO NTRATUAL, CUMULADA COM RESTITUICAO DE IMPORTANCIAS PAGAS, DANO MA TERIAL E APLICACAO DE CLAUSULAS PENAIS PROPOSTA POR CLAUDIO '
    'EDUAR DO SALEM ELIAS E ETELVINA FILOMENA CORREIA MARQUES ELIAS EM DESFA VOR DE TERRAS ALPHA ANAPOLIS EMPREENDIMENTOS IMOBILIARIOS LTDA. E VIA ANAPOLIS LTDA. '
    'INICIALMENTE DISCORREM AS PARTES AUTORAS QUE ADQUIRIRAM UM LOTE DAS RES NO EMPREENDIMENTO DENOMINADO TERRAS AL PHAVILLE ANAPOLIS, COM A PROMESSA DE QUE PODERIAM '
    'EDIFICAR EM ATE 60% DA AREA COMERCIALIZADA (TAXA DE OCUPACAO), BEM ASSIM O COEFI CIENTE DE PERMEABILIDADE SERIA DE APENAS 20%. EXPLICAM QUE, DEPOI S DE ADQUIRIREM O '
    'IMOVEL, VIERAM A SABER QUE A LEGISLACAO MUNICIP AL RESTRINGIA A TAXA DE OCUPACAO A 30%, SENDO COEFICIENTE DE PERM EABILIDADE DE NO MINIMO 30%. ADUZEM QUE, COM '
    'A RESTRICAO NA AREA DE CONSTRUCAO, SOFRERAM GRAVE PREJUIZO, CONFIGURANDO PROPAGANDA E NGANOSA DAS RES, O QUE DEVERIA SER LEVADO EM CONSIDERACAO PARA QU E A '
    'RESCISAO DO CONTRATO FOSSE JUDICIALMENTE DECRETADA. COMO CONS EQUENCIA DA RESCISAO PLEITEADA, AS RES DEVERIAM RESTITUIR AS IMPO RTANCIAS PAGAS PELOS AUTORES, '
    'ALEM DE PAGAREM AS MULTAS PREVISTAS NAS CLAUSULAS QUINZE E DEZENOVE ESTABELECIDAS NO PRE-CONTRATO DE PROMESSA DE VENDA E COMPRA, BEM ASSIM AS TAXAS CONDOMINIAL '
    'QUITA DAS E, EM RAZAO DOS TRANSTORNOS SOFRIDOS, RECEBEREM DANO MORAL. D ECISAO (FLS. 148/150) ANTECIPANDO OS EFEITOS DA TUTELA DETERMINOU A ABSTENCAO DE '
    'INCLUSAO DO NOME DOS AUTORES NOS SERVICOS DE PROT ECAO AO CREDITO. EM SEDE DE CONTESTACAO (FLS. 162/190), AS RES AP RESENTAM SUA VERSAO PARA OS FATOS E ALEGARAM '
    'QUE A TAXA DE OCUPAC AO PARA O EMPREENDIMENTO ESTA PLENAMENTE DE ACORDO COM AQUILO QUE DETERMINA O ARTIGO 21 DO PLANO DIRETOR DO MUNICIPIO DE ANAPOLIS. ... '
    'RELATARAM QUE, MESMO ANTES DE SER SANCIONADA A LEI COMPLEMENT AR MUNICIPAL N 334/2015 (QUE ALTEROU A TAXA DE OCUPACAO NO MUNICI PIO), A PREFEITURA JA ESTAVA '
    'EMITINDO ALVARA DE CONSTRUCAO COM AS NOVAS TAXAS, CONFORME E-MAIL COLACIONADO NA PETICAO DE FLS. 166. VERBERARAM AINDA QUE OS AUTORES NAO COMPROVARAM O PREJUIZO '
    'ALEGA DO, SEQUER DEMONSTRANDO A EFETIVA RECUSA DO CONDOMINIO E POSTERIO RMENTE, DA PREFEITURA COM RELACAO AOS PROJETOS, NO QUE TANGE A DI SCUSSAO SOBRE A '
    'TAXA DE OCUPACAO. NOUTRO GIRO, RESSALTAM QUE A TA XA DE CONDOMINIO E OBRIGACAO DECORRENTE DO CONTRATO ASSINADO PELO S AUTORES E, AINDA QUE NAO HAJA '
    'EDIFICACAO, DEVE SER PAGA, SENDO DESFUNDAMENTADO O REQUERIMENTO DE RESTITUICAO. LADO OUTRO, EXPLIC A QUE JAMAIS SE OPUSERAM A RESCISAO DO CONTRATO, INCLUSIVE '
    'HAVEND O, NO CONTRATO FIRMADO ENTRE AS PARTES, CLAUSULAS ESPECIFICAS A E SSE RESPEITO. ENTRETANTO, EM CASO DE ROMPIMENTO DO QUE FORA ENTAB ULADO, ENTENDEM QUE OS '
    'AUTORES DEVEM ARCAR COM OS ENCARGOS LA EST ABELECIDOS, VEZ QUE AS RES EM NADA CONTRIBUIRAM PARA O SEU DESCUM PRIMENTO. ADUZEM TAMBEM IMPOSSIBILIDADE DE '
    'CONDENACAO EM PAGAMENT O DA MULTA CONTRATUAL (CLAUSULA PENAL), BEM ASSIM DA APLICACAO DE MULTA CONTRATUAL. POR FIM, ALEGAM INEXISTIR DANO MORAL A SER '
    'IND ENIZADO. IMPUGNACAO DA CONTESTACAO JUNTADA AS FLS. 370/376. AS RE S APRESENTARAM SUAS CONSIDERACOES FINAIS AS FLS. 391/392, TENDO O S AUTORES APRESENTADO AS '
    'FLS. 398/399. POIS BEM. TEM-SE QUE OS AU TORES ADQUIRIRAM IMOVEL DAS RES, TENDO DESTAS RECEBIDO A PROMESSA DE QUE A TAXA DE OCUPACAO NO LOTE NEGOCIADO ERA DE 60%, '
    'SENDO O COEFICIENTE DE PERMEABILIDADE DE 20%. ALEGAM QUE, APOS A COMPRA, FORAM SURPREENDIDOS COM A INFORMACAO DE QUE A LEI MUNICIPAL PREVI A 30% PARA CADA UMA '
    'DAS HIPOTESES, O QUE CONSIDERARAM SER PROPAGA NDA ENGANOSA, PLEITEANDO A RESCISAO DO CONTRATO COM RESTITUICAO D OS OS VALORES ATE ENTAO PAGOS, INCLUSIVE DA TAXA '
    'DE CONDOMINIO, A LEM DAS MULTAS ALI PREVISTAS E DANO MORAL PELO TRANSTORNO SOFRIDO . PARA LASTREAR SUAS ALEGACOES, APRESENTARAM O PRE-CONTRATO DE PR OMESSA DE '
    'VENDA E COMPRA (FLS. 24), INSTRUMENTO PARTICULAR DE PRO MESSA DE COMPRA E VENDA DE UNIDADE AUTONOMA E OUTRAS AVENCAS (FLS . 34), DECISAO DO PROCON-ANAPOLIS APLICANDO '
    'MULTA AS RES (FLS. 42 ), TABELA DE CORRECAO E COMPROVANTES DE VALORES PAGOS (FLS. 54) E SENTENCAS PROFERIDAS EM SITUACOES QUE ENTENDE SER ANALOGAS AO CA SO '
    'EM TELA (FLS. 400). NO ENTANTO, NOTE-SE QUE OS AUTORES FORAM D ESIDIOSOS AO TENTAR COMPROVAR O ALEGADO. ISSO PORQUE, SEQUER FOI JUNTADO POR ELES A LEI '
    'MUNICIPAL EM QUE ESTARIAM CONTIDAS AS REST RICOES SOBRE A AREA A SER EDIFICADA NO IMOVEL ADQUIRIDO. TAMBEM N AO JUNTOU O REGULAMENTO DO CONDOMINIO '
    '(ITEM C.4, PG. 36) PRETENSA MENTE EM CONFLITO COM A CITADA LEI, APENAS ALEGANDO, SEM QUALQUER PROVA, TER SOFRIDO DANO. NEM MESMO APOS SEREM INTIMADOS '
    '(FLS. 38 9 E 390) OS AUTORES JUNTARAM DOCUMENTOS QUE COMPROVEM A NEGATIVA DA FAZENDA MUNICIPAL EM APROVAR A EDIFICACAO NO IMOVEL COM A TAXA DE OCUPACAO '
    'CONTIDA NO REGULAMENTO DO CONDOMINIO A EPOCA DA AQUI SICAO (60%, CONFORME ALEGADO E NAO PROVADO). NAO OBSTANTE, OS AUT ORES INSISTIRAM QUE HOUVE PROPAGANDA '
    'ENGANOSA E REQUERERAM A RESC ISAO CONTRATUAL (FLS. 398/399). CABE AQUI RAPIDAMENTE DISCORRER S OBRE O CONTRATO. ESTE, POSSUI COMO PRIMEIRA FUNCAO DIRIGIR OS '
    'PAC TOS PARA A CONSECUCAO DE FINALIDADES QUE ATENDAM AOS INTERESSES D A COLETIVIDADE1. SOBRE O TEMA, HA IMPORTANTE LICAO DO PROFESSOR F LAVIO TARTUCE2: '
    'O CONTRATO DEVE SER, NECESSARIAMENTE, INTERPRETAD O E VISUALIZADO DE ACORDO COM O CONTEXTO DA SOCIEDADE. A PALAVRA FUNCAO SOCIAL DEVE SER VISUALIZADA COM O '
    'SENTIDO DE FINALIDADE CO LETIVA, SENDO EFEITO DO PRINCIPIO EM QUESTAO A MITIGACAO OU RELAT IVIZACAO DA FORCA OBRIGATORIA DAS CONVENCOES '
    '(PACTA SUNT SERVANDA ). NESSE CONTEXTO, O CONTRATO NAO PODE SER MAIS VISTO COMO UMA BO LHA, QUE ISOLA AS PARTES DO MEIO SOCIAL. SIMBO '
    'LOGICAMENTE, A FUN CAO SOCIAL FUNCIONA COMO UMA AGULHA, QUE FURA A BOLHA, TRAZENDO U MA INTERPRETACAO SOCIAL DOS PACTOS. OUTRO NAO E O ENTENDIMENTO DO '
    'SUPERIOR TRIBUNAL DE JUSTICA: CIVIL. PROCESSO CIVIL. [] 1. A 5. OMISSIS. 6. O PRINCIPIO DO PACTA SUNT SERVANDA NAO CONSTITUI OBIC E A REVISAO CONTRATUAL, '
    'MORMENTE ANTE OS PRINCIPIOS DA BOA-FE OBJ ETIVA, DA FUNCAO SOCIAL QUE OS EMBALA E DO DIRIGISMO QUE OS NORTE IA. PRECEDENTES. 7. A 9. OMISSIS. '
    '(AGRG NO RESP 1363814/PR, REL. MINISTRO MOURA RIBEIRO, TERCEIRA TURMA, JULGADO EM 17/12/2015, DJ E 03/02/2016) PERCEBE-SE AQUI A NITIDA ORIENTACAO, TANTO DA DOUTR '
    'INA, QUANTO DA JURISPRUDENCIA, NO SENTIDO DE QUE AS REGRAS CONTRA TUAIS PODEM SOFRER REVISAO ANTE A NECESSIDADE DE SUA ADEQUACAO A FUNCAO SOCIAL, CONFORME PRETENDIDO '
    'PELOS AUTORES. REFERIDO ENTEND IMENTO DEMONSTRA QUE A ALTERACAO CONTRATUAL PLEITEADA PODE SER AT ENDIDA, DESDE QUE COMPROVADA A REAL NECESSIDADE DE INGERENCIA DA '
    'FORCA ESTATAL NA RELACAO PARTICULAR. OCORRE, QUE NAO FICOU DEMONS TRADO QUALQUER PREJUIZO SOFRIDO PELOS AUTORES CAPAZ DE RESCINDIR O PACTO REALIZADO ENTRE AS '
    'PARTES, O QUE, PER SI, OBSTA A CONCESS AO DO PLEITO DA INICIAL. COMO EXPLICITADO, NAO HOUVE COMPROVACAO DA INTENCAO DOS AUTORES EM CONSTRUIR NA AREA ADQUIRIDA DAS '
    'RES, N AO SENDO DEMONSTRADA A ALEGADA LIMITACAO GERADORA DE DANO. OS AUT ORES RESTRINGIRAM-SE A ALEGACOES QUE HOUVE LIMITACAO NO DIREITO D E EDIFICAR NO PERCENTUAL '
    'ANUNCIADO, SEM QUE O PREJUIZO FOSSE ATES TADO, NEM MESMO FOSSE COMPROVADO QUE O FATOR LIMITANTE OBSTOU OS AUTORES DE EDIFICAR NO LOTE COMERCIALIZADO. ISSO NAO BASTASSE, '
    'AS RES DEMONSTRARAM QUE HOUVE, POSTERIORMENTE A PROPOSITURA DA ACAO , ALTERACAO NA LEI COMPLEMENTAR MUNICIPAL QUE DISPOE SOBRE O PLAN O DIRETOR DE ANAPOLIS. JUNTARAM AS '
    'FLS. 195/206 MENCIONADA LEGISL ACAO, DEVIDAMENTE ALTERADA PELA LEI COMPLEMENTAR N334/2015 QUE, D ENTRE OUTROS DISPOSITIVOS, MODIFICA O INCISO III, DO ARTIGO 7 DO PLANO '
    'DIRETOR. COM TAL MUDANCA, O COEFICIENTE DE OCUPACAO MAXIMA POR UNIDADE AUTONOMA PASSOU DE 30% PARA 70%. PERCEBE-SE QUE A NOV IDADE LEGISLATIVA ABARCOU O CERNE DA QUESTAO '
    'LEVANTADA PELOS AUTO RES. SE ANTES, HAVIA SIDO OFERECIDO PELAS RES A POSSIBILIDADE DE UTILIZAR-SE DE 60% DA AREA AUTONOMA PARA CONSTRUCAO, ATUALMENTE O MUNICIPIO DE '
    'ANAPOLIS PERMITE UTILIZACAO DE 70% DA AREA PARA A M ESMA FINALIDADE. AINDA QUE TAL MUDANCA NAO TENHA SIDO PROMOVIDA P ELAS RES, OS AUTORES FORAM AMPLA E DIRETAMENTE '
    'BENEFICIADOS. SE C ONSIDERAR QUE EXISTIA UM DESCOMPASSO CONSIDERADO GRAVE ENTRE O PR OMETIDO PELAS RES E O PERMITIDO PELA LEGISLACAO MUNICIPAL, HOJE T EM-SE MODIFICACAO '
    'FAVORAVEL QUE INTERFERE DE MANEIRA DEFINITIVA N A LIDE, COMO DITO, FAVORECENDO OS AUTORES, SENDO QUE OS AUTORES N AO DEMOSTRARAM O DANO SUPORTADO NA FASE ANTERIOR. '
    'ANTE O EXPOSTO, REVOGO A LIMINAR CONCEDIDA AS FLS. 148/150 E REJEITO O PEDIDO DO S AUTORES, PARA JULGAR EXTINTO O PROCESSO, COM RESOLUCAO DE MERIT O, NOS TERMOS DO '
    'ARTIGO 487, INCISO I, DO CPC. CONDENO OS AUTORES AO PAGAMENTO DAS CUSTAS E DESPESAS PROCESSUAIS E NO PAGAMENTO DE 10% SOBRE O VALOR ATUALIZADO DA CAUSA A TITULO DE '
    'HONORARIOS ADV OCATICIOS, NOS TERMOS DO ARTIGO 85, 2, DO CPC. EM CASO DE INTERPO SICAO DE RECURSO DE APELACAO, INTIME-SE A PARTE APELADA PARA APRE SENTAR AS SUAS '
    'CONTRARRAZOES NO PRAZO DE 15 (QUINZE) DIAS, NOS TE RMOS DO ART. 1.010, 1, DO CODIGO DE PROCESSO CIVIL. FINDO O PRAZO , COM OU SEM AS CONTRARRAZOES, CERTIFIQUE-SE E '
    'REMETAM-SE OS AUTO S AO E. TRIBUNAL DE JUSTICA DO ESTADO DE GOIAS. NO ENTANTO, CASO SEJA INTERPOSTA APELACAO ADESIVA, INTIME-SE A PARTE APELANTE (APE LADA DO SEGUNDO '
    'RECURSO) PARA APRESENTAR AS CONTRARRAZOES, TAMBEM EM 15 (QUINZE) DIAS. EXPIRADO O PRAZO ACIMA, COM OU SEM AS CONTR ARRAZOES AO RECURSO ADESIVO, CERTIFIQUE-SE E '
    'REMATAM-SE OS AUTOS AO TRIBUNAL DE JUSTICA, NOS TERMOS DO ART. 1.010, 3, TAMBEM DO CO DIGO DE PROCESSO CIVIL. COM O TRANSITO EM JULGADO DESTA SENTENCA, E NADA '
    'REQUERENDO AS PARTES NO PRAZO DE 30 DIAS, ARQUIVEM-SE OS AUTOS COM AS CAUTELAS DE ESTILO. PUBLIQUE-SE. REGISTRE-SE. INTIME M-SE. ANAPOLIS/GO, 1 DE AGOSTO DE 2017. '
    'ELAINE CHRISTINA ALENCAST RO VEIGA ARAUJO JUIZA DE DIREITO')),

    build_deadline_case_test(('Nº 2015.04.1.009842-3 - Divorcio Consensual - A: I. A. M.e.o.. Adv(s). : GO034059 - LARISSA OLIVEIRA DUTRA, DF006479 - Divino Jose Santos. '
    'R: N. H.. Adv(s). : NAO CONSTA ADVOGADO. A: L. D. S. R. M.. Adv(s). : (.). DECISAO - Acolho a emenda à inicial de fls. 134/141. Designe-se nova data para audiência de '
    'conciliação, citando-se o requerido e encaminhando cópia da emenda supramencionada. Não havendo acordo na audiência, o prazo para oferecer defesa será de 15 '
    '(quinze) dias úteis, contados da data da audiência, independentemente do comparecimento das partes, devendo a especificação de eventuais provas ser feita na '
    'própria contestação. Apresentada contestação, intime-se a parte autora para réplica, bem como para que especifique as provas que pretende produzir. Decorrido o '
    'prazo sem contestação, após a devida certificação pela secretaria, intime-se a parte autora para especificação de provas ou para requerer o julgamento antecipado da '
    'lide. Em seguida, ao Ministério Público. Cumpridas todas as determinações precedentes, venham os autos conclusos para saneamento do processo. Gama - DF, quarta-feira, '
    '24/05/2017 às 15h56. Gildete Silva Balieiro,Juíza de Direito 2.')),

    build_deadline_case_test(('Nº 2017.03.1.003596-8 - Busca e Apreensao Em Alienacao Fiduciaria - A: BANCO RCI BRASIL SA. Adv(s). : DF036999 - Antonio Samuel da Silveira. '
    'R: RANILSON TOME DE PAIVA. Adv(s). : Nao Consta Advogado. Compulsando os autos, verifica-se que o veículo objeto da demanda foi recolhido ao Departamento de Trânsito '
    'do Distrito Federal, conforme ofício de fl. 102/105. Assim, fica o autor intimado para se manifestar sobre o ofício acima, informando se tem interesse na alienação do '
    'bem em leilão público ou para requerer outra providência apta ao prosseguimento do feito, nos termos do despacho de fl. 111. Prazo: 05 dias úteis, sob pena de '
    'extinção. Ceilândia - DF, quarta-feira, 28/06/2017 às 15h21. João Ricardo Viana Costa,Juiz de Direito Substituto .')),

    build_deadline_case_test(('NR. PROTOCOLO : 338041-33.2000.8.09.0067 ( 200003380410 ) AUTOS NR. : 450 NATUREZA : EXECUCAO EXECUTADO : ADAILSON COSTA LUCIENE MENEZES COSTA '
    'EXEQUENTE : BANCO DO BRASIL SA ADV EXECDO : 12199 GO - OSVALDO BONIFACIO JUNIOR ADV EXEQTE : 31075 GO - GUSTAVO AMATO PISSINI 33788 GO - IVAN MARCIANO DE FREITAS '
    '261030 SP - GUSTAVO AMATO PISSINI 48029 GO - VIKTOR BRUNO PEREIRA DA SILVA 44132 GO - NEI CALDERON 44131 GO - MARCELO OLIVEIRA ROCHA DESPACHO : PROCESSO N. '
    '200003380410 DECISAO DEFIRO O PEDIDO DE FL. 271. ASSI M, COM ALICERCE NO ARTIGO 854 DO CPC / 2015, DETERMINO A PENHORA ONLINE DE ATIVOS FINANCEIROS EM NOME DO(S) '
    'EXECUTADO(S), NO MONTA NTE CORRESPONDENTE AO VALOR INTEGRAL DO DEBITO DESCRITO NOS AUTOS , ACRESCIDO DE HONORARIOS (10%), UTILIZANDO-SE O CONVENIO BACENJU D, '
    'VALENDO-SE O DOCUMENTO EMITIDO PELO BANCO CENTRAL COMO AUTO DE PENHORA (854, 5, DO CPC). EFETUADO O BLOQUEIO, PROCEDA-SE A TRAN SFERENCIA PARA CONTA JUDICIAL. '
    'NA SEQUENCIA, DE-SE A CIENCIA AO(S ) EXECUTADO(S), NA FORMA RECOMENDADA PELO ARTIGO 854, 2 E 3 DO CP C / 2015. RECAINDO A CONSTRICAO SOBRE VALORES IRRISORIOS, '
    'PROCEDA -SE AO DESBLOQUEIO. APOS, DETERMINO A INTIMACAO DO EXEQUENTE PARA REQUERER O QUE FOR PERTINENTE, NO PRAZO MAXIMO DE 05 (CINCO) DIA S, SOB PENA DE EXTINCAO. '
    'CADASTRE-SE NO SPG (SISTEMA DE PRIMEIRO GRAU) OS PROCURADORES INFORMADOS A FL. 271, CONFORME REQUERIDO. P OR FIM, REMETAM OS AUTOS CONCLUSOS PARA DELIBERACAO. '
    'INTIME-SE. C UMPRA-SE. MAURILANDIA-GO, 26 DE JUNHO DE 2017. PAULO ROBERTO PALU DO JUIZ SUBSTITUTO')),

    build_deadline_case_test(('Processo Nº RTOrd-0000707-08.2017.5.10.0021 RECLAMANTE SINDICATO INTERESTADUAL DOS TRAB NAS IND MET MEC MAT ELETRICOS E ELETRONICOS DO DF GO '
    'TO ADVOGADO FERNANDO MARTINS DE FREITAS(OAB: 24144/DF) ADVOGADO FABIANA LANDIM DE FREITAS(OAB: 25856/DF) ADVOGADO RICARDO COELHO DE MEDEIROS(OAB: 21791/DF) '
    'RECLAMADO RGS LANTERNAGEM PINTURA LTDA - ME Intimado(s)/Citado(s): - SINDICATO INTERESTADUAL DOS TRAB NAS IND MET MEC MAT ELETRICOS E ELETRONICOS DO DF GO TO PODER '
    'JUDICIÁRIO JUSTIÇA DO TRABALHO CONCLUSÃO (Pje/JT) Conclusão feita pelo(a) servidor(a) ANA MAICÁ, em16 de Julho de 2017. A notificação remetida ao RECLAMADO retornou '
    'da Empresa Brasileira de Correios e Telégrafos com informação de \'endereço insuficiente\'. Intime-se oRECLAMANTE para que, no prazo de quinze dias, emende a petição '
    'inicial (CPC, Arts. 319, II e 321, parágrafo único), informando o correto endereço do RECLAMADO, inclusive com o número do CEP, sob pena de extinção do processo '
    'sem resolução do mérito. Informado o novo endereço, fica desde já determinada a retificação da autuação e a notificação do reclamado no novo endereço informado. '
    'Publique-se. BRASILIA, 16 de Julho de 2017 LUIZ HENRIQUE MARQUES DA ROCHA Juiz do Trabalho Titular')),
]

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)

# Colocando o modelo em modo de evaluation
model.eval()
model.to('cuda')

# If you have a GPU, put everything on cuda
#tokens_tensor = tokens_tensor.to('cuda')


prediction_average_time = []

with open(PREDICTION_RESULT_PATH, "w") as txt_file:

    for case_test in CASES_TEST:
        for question in case_test['questions']:
            date_inicio = datetime.now()
            txt_file.write(answer(question, case_test['text']) + '\n')
            date_fim = datetime.now()
            diff = (date_fim - date_inicio).total_seconds() * 1000
            #print(diff)

            # Tempo de previsão em milisegundos
            prediction_average_time.append(diff)

        txt_file.write('\n')
        #print("\n")

print("\n")
print("Tempo médio de predição: {} ms".format( sum(prediction_average_time) / len(prediction_average_time) ))