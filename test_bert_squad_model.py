import os, torch
from transformers import BertTokenizer, BertForQuestionAnswering
from datetime import datetime

MODEL_PATH = 'D:\\Github\\trts_crawler\\1.1\\corpus server\\trained_benchmark_case_100_cento'

def build_case_test(text, question_array):
    return {
        'text': text,
        'questions': question_array
    }

def answer(question, text):
    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
    input_ids = tokenizer.encode(input_text)

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
        return ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace(" ##", "")

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
    
    build_case_test(('Numeração única: 28139-54.2015.4.01.3400\n28139-54.2015.4.01.3400 AÇÃO ORDINÁRIA / OUTRAS\nAUTOR :   ITAU UNIBANCO S.A.\nADVOGADO :   SP00198407 - '
    'DIOGO PAIVA MAGALHAES VENTURA\nREU :   UNIAO FEDERAL\nO Exmo. Sr. Juiz exarou :\n\" Vista ao autor para manifestar-se no prazo de 10 (dez) dias acerca dos embargos\n'
    'declaratórios opostos pela União, fls. 240/242, tendo em vista seus possíveis efeitos\nmodificativos.\"'),
    [
        'Qual o prazo?',
        'O autor deverá manifestar-se em quantos dias?',
    ]),

    build_case_test(('AS FLS.44/46 O EXEQUENTE REQUEREU A PENHORA SOBRE OS\nDIREITOS DO EXECUTADO INCIDENTES SOBRE O VEICULO LOCALIZADO. EM\nNOVA PESQUISA ATRAVES DO '
    'SISTEMA RENAJUD, CONSTATOU-SE QUE O VEIC\nULO FOI ALIENADO PARA RAFAEL THOMAZINI, FL.48. INSTADO A MANIFEST\nAR-SE, O EXEQUENTE ARGUMENTOU QUE OCORREU FRAUDE A EXECUCAO '
    'E REQ\nUEREU A DECLARACAO DE INEFICACIA DA ALIENACAO DO VEICULO, FLS.52/\n53. DESPACHO DE FL.56 DETERMINOU A INTIMACAO DO EXECUTADO, NAO SE\nNDO ESTE ENCONTRADO NO '
    'ENDERECO INFORMADO NOS AUTOS, CONFORME AVI\nSO DE RECEBIMENTO DE FL.59. EM PETICAO DE FLS.81/82 A EXEQUENTE R\nEQUEREU A REALIZACAO DE DILIGENCIAS JUNTO AO SISTEMA '
    'BACENJUD, A\nFIM DE LOCALIZAR O ENDERECO ATUALIZADO DO EXECUTADO. CUMPRE SALIE\nNTAR QUE E DEVER DA PARTE MANTER SEU ENDERECO ATUALIZADO, DEVENDO\nARCAR COM O ONUS DE '
    'SUA OMISSAO. ASSIM, A PARTIR DO MOMENTO EM Q\nUE O EXECUTADO FOI CITADO, CABERIA A ESTE MANTER ESTE JUIZO INFOR\nMADO DE SEU ENDERECO. NESSE PASSO, DETERMINO A EXPEDICAO '
    'DE MANDA\nDO DE INTIMACAO PARA O EXECUTADO, A SER CUMPRIDO NO ENDERECO INFO\nRMADO NOS AUTOS, A FIM DE DAR CUMPRIMENTO AO DESPACHO DE FL.56. D\nILIGENCIE-SE JUNTO AO '
    'DETRAN/GO, EXPEDINDO-SE OFICIO CASO SE FACA\nNECESSARIO, A FIM DE OBTER INFORMACAO A RESPEITO DA TRANSFERENCI\nA DO VEICULO GM/S10 EXECUTIVE D, ANO 2008, PLACA NLU '
    '0390. COM A\nRESPOSTA, INTIME-SE O EXEQUENTE PARA MANIFESTAR-SE NO PRAZO DE CI\nNCO DIAS, DEVENDO INFORMAR O ENDERECO DO TERCEIRO ADQUIRENTE, A F\nIM DE QUE O MESMO '
    'SEJA INTIMADO, CONFORME DETERMINA O ART.792, 4,\nDO CPC'),
    [
        'Qual o prazo?',
        'O autor deverá manifestar-se em quantos dias?',
    ])
]

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)

# Colocando o modelo em modo de evaluation
model.eval()
model.to('cuda')

# If you have a GPU, put everything on cuda
#tokens_tensor = tokens_tensor.to('cuda')


prediction_average_time = []

with open(MODEL_PATH + '\\prediction_result.txt', "w") as txt_file:

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