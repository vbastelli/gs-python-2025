import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import heapq
from collections import defaultdict, deque

# ====================== CLASSES E ESTRUTURAS DE DADOS ======================

@dataclass
class Drone:
    """Classe para representar um drone de combate a incêndios"""
    id: int
    x: float
    y: float
    bateria: int
    capacidade: float
    velocidade: float
    status: str = "disponivel"  # disponivel, em_missao, manutencao
    historico_missoes: List = None
    combustivel_gasto: float = 0.0
    
    def __post_init__(self):
        if self.historico_missoes is None:
            self.historico_missoes = []

@dataclass
class FocoIncendio:
    """Classe para representar um foco de incêndio"""
    id: int
    x: float
    y: float
    intensidade: int
    vento: float
    prioridade: int
    area_afetada: float
    timestamp: datetime
    status: str = "ativo"  # ativo, controlado, extinto
    recursos_necessarios: float = 0.0
    tempo_estimado: int = 0
    
    def __post_init__(self):
        # Calcula recursos necessários baseado na intensidade e área
        self.recursos_necessarios = self.intensidade * self.area_afetada * 0.1
        self.tempo_estimado = int(self.intensidade * 5 + self.area_afetada * 2)

class FilaPrioridade:
    """Fila de prioridade para gerenciar focos de incêndio"""
    def __init__(self):
        self.heap = []
        self.contador = 0
    
    def inserir(self, foco: FocoIncendio):
        """Insere foco na fila com prioridade baseada em critérios múltiplos"""
        # Prioridade = intensidade * prioridade + área afetada - tempo desde detecção
        tempo_decorrido = (datetime.now() - foco.timestamp).total_seconds() / 3600
        prioridade = -(foco.intensidade * foco.prioridade + foco.area_afetada - tempo_decorrido)
        heapq.heappush(self.heap, (prioridade, self.contador, foco))
        self.contador += 1
    
    def extrair_proximo(self) -> Optional[FocoIncendio]:
        """Extrai o foco de maior prioridade"""
        if self.heap:
            return heapq.heappop(self.heap)[2]
        return None
    
    def tamanho(self) -> int:
        return len(self.heap)

class SistemaQueimadas:
    """Sistema principal de gerenciamento de queimadas"""
    
    def __init__(self, num_drones: int = 6, area_monitoramento: Tuple[int, int] = (50, 50)):
        self.drones: List[Drone] = []
        self.focos_ativos: FilaPrioridade = FilaPrioridade()
        self.focos_historico: List[FocoIncendio] = []
        self.area_x, self.area_y = area_monitoramento
        self.historico_operacoes: List[Dict] = []
        self.estatisticas: Dict = defaultdict(int)
        self.cache_dp = {}
        
        # Inicializar drones
        self._inicializar_drones(num_drones)
        
        # Inicializar alguns focos para demonstração
        self._gerar_focos_iniciais(8)
    
    def _inicializar_drones(self, num_drones: int):
        """Inicializa a frota de drones com características variadas"""
        random.seed(42)
        for i in range(num_drones):
            drone = Drone(
                id=i,
                x=random.uniform(0, self.area_x),
                y=random.uniform(0, self.area_y),
                bateria=random.randint(70, 100),
                capacidade=round(random.uniform(1.5, 3.0), 2),
                velocidade=round(random.uniform(20, 40), 2)  # km/h
            )
            self.drones.append(drone)
    
    def _gerar_focos_iniciais(self, num_focos: int):
        """Gera focos iniciais para demonstração"""
        for i in range(num_focos):
            foco = FocoIncendio(
                id=i,
                x=random.uniform(0, self.area_x),
                y=random.uniform(0, self.area_y),
                intensidade=random.randint(1, 10),
                vento=round(random.uniform(0.5, 2.0), 2),
                prioridade=random.choice([1, 2, 3]),  # 1=baixa, 2=média, 3=alta
                area_afetada=round(random.uniform(0.5, 5.0), 2),
                timestamp=datetime.now() - timedelta(minutes=random.randint(0, 120))
            )
            self.focos_ativos.inserir(foco)
    
    def inserir_nova_ocorrencia(self, x: float, y: float, intensidade: int, 
                              prioridade: int = 1, area_afetada: float = 1.0):
        """Insere uma nova ocorrência de incêndio no sistema"""
        foco = FocoIncendio(
            id=len(self.focos_historico) + self.focos_ativos.tamanho(),
            x=x, y=y, intensidade=intensidade, prioridade=prioridade,
            vento=round(random.uniform(0.5, 2.0), 2),
            area_afetada=area_afetada,
            timestamp=datetime.now()
        )
        
        self.focos_ativos.inserir(foco)
        self.estatisticas['focos_detectados'] += 1
        
        print(f"✓ Nova ocorrência inserida: Foco {foco.id} em ({x:.1f}, {y:.1f})")
        print(f"  Intensidade: {intensidade}, Prioridade: {prioridade}, Área: {area_afetada:.1f} ha")
    
    def calcular_custo_completo(self, drone: Drone, foco: FocoIncendio) -> Dict:
        """Calcula custo detalhado para alocação drone-foco"""
        # Distância euclidiana
        distancia = np.sqrt((drone.x - foco.x)**2 + (drone.y - foco.y)**2)
        
        # Tempo de viagem
        tempo_viagem = distancia / drone.velocidade
        
        # Custo de combustível baseado na distância e capacidade do drone
        custo_combustivel = distancia * (2.0 - drone.capacidade/3.0)
        
        # Eficiência baseada na capacidade vs necessidade do foco
        eficiencia = min(drone.capacidade / foco.recursos_necessarios, 1.0)
        
        # Penalidades e bonificações
        penalidade_vento = foco.vento * 0.5
        bonus_prioridade = 1.0 / foco.prioridade if foco.prioridade > 0 else 1.0
        penalidade_tempo = (datetime.now() - foco.timestamp).total_seconds() / 3600 * 0.1
        
        # Custo total
        custo_base = distancia * foco.intensidade / drone.capacidade
        custo_total = (custo_base + penalidade_vento + penalidade_tempo) * bonus_prioridade
        
        # Verificar se o drone tem bateria suficiente
        bateria_necessaria = tempo_viagem * 10 + foco.tempo_estimado * 2
        viavel = drone.bateria >= bateria_necessaria and drone.status == "disponivel"
        
        return {
            'custo_total': round(custo_total, 2),
            'distancia': round(distancia, 2),
            'tempo_viagem': round(tempo_viagem, 2),
            'custo_combustivel': round(custo_combustivel, 2),
            'eficiencia': round(eficiencia, 2),
            'bateria_necessaria': round(bateria_necessaria, 2),
            'viavel': viavel
        }
    
    @lru_cache(maxsize=1000)
    def dp_alocacao_otima(self, focos_bits: int, drones_disponiveis: Tuple) -> Tuple[float, Tuple]:
        """
        Programação Dinâmica com memoização para alocação ótima
        
        Args:
            focos_bits: Representação em bits dos focos já alocados
            drones_disponiveis: Tupla com IDs dos drones disponíveis
        
        Returns:
            Tupla (custo_minimo, alocacoes)
        """
        if not drones_disponiveis:
            return 0.0, ()
        
        focos_ativos_lista = [f[2] for f in self.focos_ativos.heap if f[2].status == "ativo"]
        
        if not focos_ativos_lista or focos_bits == (1 << len(focos_ativos_lista)) - 1:
            return 0.0, ()
        
        drone_id = drones_disponiveis[0]
        drone = self.drones[drone_id]
        restantes_drones = drones_disponiveis[1:]
        
        melhor_custo = float('inf')
        melhor_alocacao = ()
        
        # Opção 1: Drone não faz nada
        custo_sem_acao, alocacao_sem_acao = self.dp_alocacao_otima(focos_bits, restantes_drones)
        if custo_sem_acao < melhor_custo:
            melhor_custo = custo_sem_acao
            melhor_alocacao = alocacao_sem_acao
        
        # Opção 2: Drone é alocado para algum foco
        for i, foco in enumerate(focos_ativos_lista):
            if not (focos_bits & (1 << i)):  # Foco ainda não alocado
                custo_info = self.calcular_custo_completo(drone, foco)
                
                if custo_info['viavel']:
                    novo_focos_bits = focos_bits | (1 << i)
                    custo_restante, alocacao_restante = self.dp_alocacao_otima(
                        novo_focos_bits, restantes_drones
                    )
                    
                    custo_total = custo_info['custo_total'] + custo_restante
                    
                    if custo_total < melhor_custo:
                        melhor_custo = custo_total
                        melhor_alocacao = ((drone_id, i, custo_info),) + alocacao_restante
        
        return melhor_custo, melhor_alocacao
    
    def executar_alocacao_otima(self) -> Dict:
        """Executa a alocação ótima usando programação dinâmica"""
        print("🔄 Calculando alocação ótima usando Dynamic Programming...")
        
        # Limpar cache para nova execução
        self.dp_alocacao_otima.cache_clear()
        
        # Drones disponíveis
        drones_disponiveis = tuple(d.id for d in self.drones if d.status == "disponivel")
        
        if not drones_disponiveis:
            return {'erro': 'Nenhum drone disponível'}
        
        inicio = time.time()
        custo_total, alocacoes = self.dp_alocacao_otima(0, drones_disponiveis)
        tempo_calculo = time.time() - inicio
        
        # Processar alocações
        focos_ativos_lista = [f[2] for f in self.focos_ativos.heap if f[2].status == "ativo"]
        alocacoes_detalhadas = []
        
        for drone_id, foco_idx, custo_info in alocacoes:
            if foco_idx < len(focos_ativos_lista):
                foco = focos_ativos_lista[foco_idx]
                drone = self.drones[drone_id]
                
                # Atualizar status
                drone.status = "em_missao"
                drone.combustivel_gasto += custo_info['custo_combustivel']
                drone.historico_missoes.append({
                    'foco_id': foco.id,
                    'timestamp': datetime.now(),
                    'custo': custo_info['custo_total']
                })
                
                alocacoes_detalhadas.append({
                    'drone_id': drone_id,
                    'foco_id': foco.id,
                    'custo_info': custo_info,
                    'foco': foco
                })
        
        # Registrar operação
        operacao = {
            'timestamp': datetime.now(),
            'tipo': 'alocacao_otima',
            'custo_total': custo_total,
            'tempo_calculo': tempo_calculo,
            'alocacoes': len(alocacoes_detalhadas),
            'cache_hits': self.dp_alocacao_otima.cache_info().hits,
            'cache_misses': self.dp_alocacao_otima.cache_info().misses
        }
        self.historico_operacoes.append(operacao)
        
        return {
            'custo_total': custo_total,
            'tempo_calculo': tempo_calculo,
            'alocacoes': alocacoes_detalhadas,
            'cache_info': self.dp_alocacao_otima.cache_info()
        }
    
    def simular_chamadas_aleatorias(self, num_chamadas: int = 5):
        """Simula chamadas aleatórias com severidade crescente"""
        print(f"🔥 Simulando {num_chamadas} chamadas aleatórias...")
        
        severidade_base = 1
        for i in range(num_chamadas):
            # Severidade crescente com alguma aleatoriedade
            severidade = min(severidade_base + random.randint(0, 2), 10)
            severidade_base += 1
            
            x = random.uniform(0, self.area_x)
            y = random.uniform(0, self.area_y)
            prioridade = min(severidade // 3 + 1, 3)
            area = random.uniform(0.5, severidade * 0.8)
            
            self.inserir_nova_ocorrencia(x, y, severidade, prioridade, area)
            
            # Simular algum tempo entre chamadas
            time.sleep(0.1)
    
    def atender_proxima_ocorrencia(self):
        """Atende a próxima ocorrência com maior prioridade"""
        foco = self.focos_ativos.extrair_proximo()
        if not foco:
            print("❌ Não há ocorrências pendentes")
            return
        
        print(f"🚁 Atendendo foco {foco.id} (Prioridade: {foco.prioridade}, Intensidade: {foco.intensidade})")
        
        # Simular atendimento
        foco.status = "controlado"
        self.focos_historico.append(foco)
        self.estatisticas['focos_atendidos'] += 1
        
        # Liberar drones que estavam nesta missão
        for drone in self.drones:
            if drone.status == "em_missao" and any(m.get('foco_id') == foco.id for m in drone.historico_missoes):
                drone.status = "disponivel"
                drone.bateria = max(20, drone.bateria - 15)  # Simular gasto de bateria
    
    def gerar_relatorio_completo(self) -> Dict:
        """Gera relatório completo das operações"""
        relatorio = {
            'timestamp': datetime.now(),
            'estatisticas_gerais': dict(self.estatisticas),
            'drones': {
                'total': len(self.drones),
                'disponiveis': sum(1 for d in self.drones if d.status == "disponivel"),
                'em_missao': sum(1 for d in self.drones if d.status == "em_missao"),
                'manutencao': sum(1 for d in self.drones if d.status == "manutencao")
            },
            'focos': {
                'ativos': self.focos_ativos.tamanho(),
                'historico': len(self.focos_historico),
                'controlados': sum(1 for f in self.focos_historico if f.status == "controlado")
            },
            'operacoes_realizadas': len(self.historico_operacoes),
            'eficiencia_dp': self.dp_alocacao_otima.cache_info()._asdict() if hasattr(self.dp_alocacao_otima, 'cache_info') else {}
        }
        
        return relatorio
    
    def visualizar_sistema_completo(self):
        """Visualização avançada do sistema completo"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Mapa principal com drones e focos
        self._plot_mapa_principal(ax1)
        
        # Subplot 2: Status dos drones
        self._plot_status_drones(ax2)
        
        # Subplot 3: Estatísticas de focos
        self._plot_estatisticas_focos(ax3)
        
        # Subplot 4: Histórico de operações
        self._plot_historico_operacoes(ax4)
        
        plt.tight_layout()
        plt.suptitle("Sistema de Combate a Queimadas - Dashboard Completo", y=1.02, fontsize=16)
        plt.show()
    
    def _plot_mapa_principal(self, ax):
        """Plot do mapa principal"""
        # Drones
        for drone in self.drones:
            cor = {'disponivel': 'blue', 'em_missao': 'orange', 'manutencao': 'red'}[drone.status]
            ax.scatter(drone.x, drone.y, c=cor, s=100, marker='^', alpha=0.7)
            ax.text(drone.x + 1, drone.y, f"D{drone.id}\n{drone.bateria}%", fontsize=8)
        
        # Focos ativos
        focos_lista = [f[2] for f in self.focos_ativos.heap]
        for foco in focos_lista:
            tamanho = foco.intensidade * 20
            cor = {1: 'yellow', 2: 'orange', 3: 'red'}[foco.prioridade]
            ax.scatter(foco.x, foco.y, c=cor, s=tamanho, alpha=0.6)
            ax.text(foco.x + 1, foco.y, f"F{foco.id}\nI:{foco.intensidade}", fontsize=8)
        
        ax.set_xlim(0, self.area_x)
        ax.set_ylim(0, self.area_y)
        ax.set_title("Mapa de Operações")
        ax.grid(True, alpha=0.3)
        ax.legend(['Drone Disponível', 'Drone em Missão', 'Drone Manutenção', 
                  'Foco Baixa Prioridade', 'Foco Média Prioridade', 'Foco Alta Prioridade'])
    
    def _plot_status_drones(self, ax):
        """Plot do status dos drones"""
        status_count = defaultdict(int)
        for drone in self.drones:
            status_count[drone.status] += 1
        
        cores = {'disponivel': 'green', 'em_missao': 'orange', 'manutencao': 'red'}
        ax.pie(status_count.values(), labels=status_count.keys(), 
               colors=[cores[k] for k in status_count.keys()], autopct='%1.1f%%')
        ax.set_title("Status da Frota de Drones")
    
    def _plot_estatisticas_focos(self, ax):
        """Plot das estatísticas de focos"""
        categorias = ['Detectados', 'Ativos', 'Controlados']
        valores = [
            self.estatisticas['focos_detectados'],
            self.focos_ativos.tamanho(),
            self.estatisticas['focos_atendidos']
        ]
        
        bars = ax.bar(categorias, valores, color=['lightblue', 'orange', 'green'])
        ax.set_title("Estatísticas de Focos de Incêndio")
        ax.set_ylabel("Quantidade")
        
        # Adicionar valores nas barras
        for bar, valor in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(valor), ha='center', va='bottom')
    
    def _plot_historico_operacoes(self, ax):
        """Plot do histórico de operações"""
        if not self.historico_operacoes:
            ax.text(0.5, 0.5, "Nenhuma operação registrada", ha='center', va='center')
            ax.set_title("Histórico de Operações")
            return
        
        tempos = [op['tempo_calculo'] for op in self.historico_operacoes if 'tempo_calculo' in op]
        custos = [op['custo_total'] for op in self.historico_operacoes if 'custo_total' in op]
        
        if tempos and custos:
            ax.scatter(tempos, custos, alpha=0.7)
            ax.set_xlabel("Tempo de Cálculo (s)")
            ax.set_ylabel("Custo Total")
            ax.set_title("Eficiência vs Custo das Operações")
        else:
            ax.text(0.5, 0.5, "Dados insuficientes", ha='center', va='center')
            ax.set_title("Histórico de Operações")
    
    def aleatorizar_drones_disponiveis(self) -> int:
        """Aleatoriza o status dos drones e retorna o número de drones disponíveis."""
        possiveis_status = ["disponivel", "em_missao", "manutencao"]
        
        # Aleatorizar o status de cada drone
        for drone in self.drones:
            drone.status = random.choice(possiveis_status)
        
        # Contar drones disponíveis
        drones_disponiveis = sum(1 for drone in self.drones if drone.status == "disponivel")
        
        print(f"🔢 Após aleatorização, {drones_disponiveis} drones estão disponíveis.")
        print("\n📋 Status de cada drone:")
        for drone in self.drones:
            print(f"   • Drone {drone.id}: {drone.status}")
        
        return drones_disponiveis

# ====================== FUNÇÃO PRINCIPAL DEMONSTRATIVA ======================

def demonstracao_sistema():
    """Demonstração completa do sistema"""
    print("=" * 80)
    print("🔥 SISTEMA AVANÇADO DE COMBATE A QUEIMADAS 🔥")
    print("=" * 80)
    
    # Inicializar sistema
    sistema = SistemaQueimadas(num_drones=6, area_monitoramento=(30, 30))
    
    print("\n0. 🎲 ALEATORIZANDO STATUS DOS DRONES")
    print("-" * 40)
    num_disponiveis = sistema.aleatorizar_drones_disponiveis()
    print(f"✅ Resultado: {num_disponiveis} drones disponíveis.")
    
    print("\n1. 📊 ESTADO INICIAL DO SISTEMA")
    print("-" * 40)
    relatorio = sistema.gerar_relatorio_completo()
    print(f"Drones disponíveis: {relatorio['drones']['disponiveis']}")
    print(f"Focos ativos: {relatorio['focos']['ativos']}")
    
    print("\n2. 🆕 INSERINDO NOVAS OCORRÊNCIAS")
    print("-" * 40)
    sistema.inserir_nova_ocorrencia(15.5, 20.3, 8, prioridade=3, area_afetada=3.2)
    sistema.inserir_nova_ocorrencia(25.1, 10.7, 6, prioridade=2, area_afetada=2.1)
    
    print("\n3. 🎲 SIMULANDO CHAMADAS ALEATÓRIAS")
    print("-" * 40)
    sistema.simular_chamadas_aleatorias(4)
    
    print("\n4. 🧠 EXECUTANDO ALOCAÇÃO ÓTIMA (DYNAMIC PROGRAMMING)")
    print("-" * 40)
    resultado = sistema.executar_alocacao_otima()
    
    if 'erro' not in resultado:
        print(f"✅ Alocação calculada em {resultado['tempo_calculo']:.4f}s")
        print(f"💰 Custo total otimizado: {resultado['custo_total']:.2f}")
        print(f"🎯 {len(resultado['alocacoes'])} alocações realizadas")
        print(f"💾 Cache DP: {resultado['cache_info'].hits} hits, {resultado['cache_info'].misses} misses")
        
        print("\n   📋 Detalhes das Alocações:")
        for alocacao in resultado['alocacoes']:
            info = alocacao['custo_info']
            print(f"   • Drone {alocacao['drone_id']} → Foco {alocacao['foco_id']}")
            print(f"     Distância: {info['distancia']:.1f}km, Custo: {info['custo_total']:.2f}")
            print(f"     Eficiência: {info['eficiencia']:.2f}, Bateria: {info['bateria_necessaria']:.1f}%")
    else:
        print(f"❌ {resultado['erro']}")
    
    print("\n5. 🚨 ATENDENDO PRÓXIMAS OCORRÊNCIAS")
    print("-" * 40)
    for i in range(3):
        sistema.atender_proxima_ocorrencia()
    
    print("\n6. 📊 RELATÓRIO FINAL")
    print("-" * 40)
    relatorio_final = sistema.gerar_relatorio_completo()
    print(f"📈 Estatísticas Finais:")
    for key, value in relatorio_final['estatisticas_gerais'].items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🚁 Status da Frota:")
    for status, qtd in relatorio_final['drones'].items():
        if status != 'total':
            print(f"   • {status.replace('_', ' ').title()}: {qtd}")
    
    print(f"\n🔥 Status dos Focos:")
    for status, qtd in relatorio_final['focos'].items():
        print(f"   • {status.replace('_', ' ').title()}: {qtd}")
    
    print(f"\n⚡ Operações Realizadas: {relatorio_final['operacoes_realizadas']}")
    
    print("\n7. 📊 VISUALIZAÇÃO DO SISTEMA")
    print("-" * 40)
    print("Gerando dashboard completo...")
    sistema.visualizar_sistema_completo()
    
    return sistema

# ====================== EXECUÇÃO ======================

if __name__ == "__main__":
    sistema = demonstracao_sistema()
    
    print("\n" + "=" * 80)
    print("🎯 DEMONSTRAÇÃO CONCLUÍDA!")
    print("💡 Principais características implementadas:")
    print("   • ✅ Programação Dinâmica com memoização (@lru_cache)")
    print("   • ✅ Funções recursivas para otimização")
    print("   • ✅ Estruturas de dados avançadas (heap, filas)")
    print("   • ✅ Sistema completo de gerenciamento")
    print("   • ✅ Visualizações e relatórios detalhados")
    print("   • ✅ Simulação de cenários realistas")
    print("   • ✅ Aleatorização de drones disponíveis")
    print("=" * 80)