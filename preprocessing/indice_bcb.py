BCB = {}

BCB['DOLAR_VENDA'] = [1, "Dólar (venda)"]
BCB['DOLAR_COMPRA'] = [10813, "Dólar (compra)"]
BCB['EURO_VENDA'] = [21619, "Euro (venda)"]
BCB['EURO_COMPRA'] = [21620, "Euro (compra)"]
BCB['IPCA_GASOLINA'] = [4458, "IPCA (gasolina)"]
BCB['GASTO_TI'] = [2769, "GASTOS COM TI"]
BCB['ICCRS'] = [12544, "ICC em REAIS"]
BCB['IGPM'] = [189, "IGP-M"]
BCB['OURO_BMF'] = [4, "Ouro BMF"]
BCB['SELIC'] = [11, "Selic"]
BCB['ICC'] = [4393, "ICC"]
BCB['DESEMPREGO'] = [24369, "Desemprego"]
BCB['VENDAS_AUTOPECAS_SUDESTE'] = [13607, "VENDAS de Automóveis e peças no sudeste"]
BCB['VENDAS_MOTOCICLOS'] = [1381, "Vendas de motociclos"]
BCB['VENDAS_PARTES_PECAS_AUTOMOVEIS'] = [1548, "Índice volume de vendas no varejo - Automóveis, motocicletas, partes e peças - Brasil"]
BCB['PIB'] = [4380, "PIB mensal - Valores correntes (R$ milhões)"]
BCB['PRODUCAO_ACO'] = [7357, "Produção Aço Bruto"]
BCB['IPCA'] = [433, "IPCA"]


def get_value(indice):
    return BCB[indice][0]


def get_reference(indice):
    return 'BCB/' + str(get_value(indice))


def get_description(indice):
    return BCB[indice][1]


def list_to_cod(indice):
   return ['BCB/' + str(BCB[i][0]) for i in indice]