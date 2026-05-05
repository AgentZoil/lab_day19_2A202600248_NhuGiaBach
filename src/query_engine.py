import networkx as nx
import re
from typing import List, Dict, Tuple, Optional

class QueryEngine:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def find_entity_in_query(self, query: str) -> str:
        """Tìm entity trong câu hỏi"""
        query_lower = query.lower()
        # Ưu tiên các entity dài hơn trước để tránh nhầm lẫn
        sorted_nodes = sorted(self.graph.nodes(), key=len, reverse=True)
        for node in sorted_nodes:
            if node.lower() in query_lower:
                return node
        return None
    
    def answer(self, query: str, hops: int = 2) -> str:
        """Trả lời câu hỏi dựa trên đồ thị"""
        query_lower = query.lower()

        multi_hop_answer = self._answer_multi_hop(query_lower, hops)
        if multi_hop_answer:
            return multi_hop_answer

        entity = self.find_entity_in_query(query)
        if not entity:
            return f"❌ Không tìm thấy thông tin liên quan đến câu hỏi: '{query}'"
        
        context = self._get_context(entity, hops)
        
        if not context:
            return f"❌ Không có thông tin kết nối nào cho '{entity}'"

        return self._synthesize_answer(query_lower, entity, context)

    def _synthesize_answer(self, query_lower: str, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Turn retrieved graph context into the final answer."""
        if any(word in query_lower for word in ['when was', 'year', 'năm thành lập', 'thành lập năm']):
            return self._answer_year(entity, context)

        if any(word in query_lower for word in ['founder', 'founded', 'founded by', 'thành lập bởi', 'ai đã thành lập']):
            return self._answer_founder(entity, context)

        if any(word in query_lower for word in ['ceo', 'giám đốc', 'tổng giám đốc']):
            return self._answer_ceo(entity, context)

        if any(word in query_lower for word in ['headquarter', 'based in', 'located', 'trụ sở', 'đặt tại']):
            return self._answer_location(entity, context)

        if any(word in query_lower for word in ['parent', 'công ty mẹ']):
            return self._answer_parent(entity, context)

        # Trả về toàn bộ evidence nếu không xác định được loại câu hỏi.
        return self._format_context(entity, context)

    def _answer_multi_hop(self, query_lower: str, hops: int) -> Optional[str]:
        """Handle a few common multi-hop question patterns."""
        founder_match = re.search(r'company (?:was )?founded by ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)
        ceo_match = re.search(r'company whose ceo is ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)
        founder_name_match = re.search(r'founder of the company (?:was )?founded by ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)
        who_founded_company_match = re.search(r'who founded the company (?:led by|whose ceo is) ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)
        company_of_ceo_match = re.search(r'which company does ([a-zA-Z][a-zA-Z\s.\'-]+) (?:lead|run)', query_lower)
        hq_city_match = re.search(r'headquarters? city of the company (?:was )?founded by ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)
        founded_year_match = re.search(r'founding year of the company (?:was )?founded by ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)
        founders_of_company_match = re.search(r'founders? of the company (?:led by|whose ceo is) ([a-zA-Z][a-zA-Z\s.\'-]+)', query_lower)

        if founder_match and 'ceo' in query_lower:
            founder = founder_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                context = self._get_context(company, hops)
                return self._answer_ceo(company, context)

        if founder_match and any(word in query_lower for word in ['headquarter', 'based in', 'located', 'trụ sở', 'đặt tại']):
            founder = founder_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                context = self._get_context(company, hops)
                return self._answer_location(company, context)

        if founder_match and any(word in query_lower for word in ['parent company', 'công ty mẹ']):
            founder = founder_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                context = self._get_context(company, hops)
                return self._answer_parent(company, context)

        if founder_match and not any(word in query_lower for word in ['ceo', 'headquarter', 'based in', 'located', 'parent company', 'year', 'founding']):
            founder = founder_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                return f"✅ {company}"

        if ceo_match and any(word in query_lower for word in ['founded', 'thành lập']):
            ceo = ceo_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("CEO", ceo)
            if company:
                context = self._get_context(company, hops)
                return self._answer_founder(company, context)

        if founder_name_match and any(word in query_lower for word in ['ceo', 'who is the ceo']):
            founder = founder_name_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                context = self._get_context(company, hops)
                return self._answer_ceo(company, context)

        if company_of_ceo_match:
            ceo = company_of_ceo_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("CEO", ceo)
            if company:
                return f"✅ {ceo} leads {company}"

        if founders_of_company_match:
            ceo = founders_of_company_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("CEO", ceo)
            if company:
                context = self._get_context(company, hops)
                return self._answer_founder(company, context)

        if who_founded_company_match:
            ceo = who_founded_company_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("CEO", ceo)
            if company:
                context = self._get_context(company, hops)
                return self._answer_founder(company, context)

        if hq_city_match:
            founder = hq_city_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                context = self._get_context(company, hops)
                return self._answer_location(company, context)

        if founded_year_match:
            founder = founded_year_match.group(1).strip().title().rstrip("?.,")
            company = self._find_company_by_relation("FOUNDED_BY", founder)
            if company:
                context = self._get_context(company, hops)
                return self._answer_year(company, context)

        return None

    def _find_company_by_relation(self, relation: str, target: str) -> Optional[str]:
        """Find the company node that points to target via relation."""
        target_lower = target.lower()
        for subj, obj, data in self.graph.edges(data=True):
            if data.get('relation') == relation and obj.lower() == target_lower:
                return subj
        return None
    
    def _get_context(self, entity: str, hops: int) -> List[Tuple[str, str, str]]:
        """Lấy context trong phạm vi hops"""
        result = []
        visited = {entity}
        current = [entity]
        
        for _ in range(hops):
            next_nodes = []
            for node in current:
                # Outgoing edges
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        rel = self.graph[node][neighbor]['relation']
                        result.append((node, rel, neighbor))
                        visited.add(neighbor)
                        next_nodes.append(neighbor)
                
                # Incoming edges
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        rel = self.graph[neighbor][node]['relation']
                        result.append((neighbor, rel, node))
                        visited.add(neighbor)
                        next_nodes.append(neighbor)
            
            current = next_nodes
        
        return result
    
    def _answer_founder(self, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Trả lời câu hỏi về người sáng lập"""
        founders = []
        for subj, rel, obj in context:
            if rel == "FOUNDED_BY" and subj == entity:
                founders.append(obj)
        if founders:
            return f"✅ {entity} was founded by: {', '.join(founders)}"
        return f"❌ Cannot find founder information for {entity}"
    
    def _answer_ceo(self, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Trả lời câu hỏi về CEO"""
        for subj, rel, obj in context:
            if rel == "CEO" and subj == entity:
                return f"✅ The current CEO of {entity} is {obj}"
        return f"❌ Cannot find CEO information for {entity}"
    
    def _answer_location(self, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Trả lời câu hỏi về trụ sở"""
        for subj, rel, obj in context:
            if rel == "HEADQUARTERED_IN" and subj == entity:
                return f"✅ {entity} is headquartered in {obj}"
        return f"❌ Cannot find headquarters information for {entity}"
    
    def _answer_year(self, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Trả lời câu hỏi về năm thành lập"""
        for subj, rel, obj in context:
            if rel == "FOUNDED_IN" and subj == entity:
                return f"✅ {entity} was founded in {obj}"
        return f"❌ Cannot find founding year information for {entity}"
    
    def _answer_parent(self, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Trả lời câu hỏi về công ty mẹ"""
        for subj, rel, obj in context:
            if rel == "PARENT_COMPANY" and subj == entity:
                return f"✅ {entity}'s parent company is {obj}"
        return f"❌ Cannot find parent company information for {entity}"
    
    def _format_context(self, entity: str, context: List[Tuple[str, str, str]]) -> str:
        """Format context thành text"""
        if not context:
            return f"📊 No connected information found for {entity}"
        
        lines = [f"📊 Information about {entity}:"]
        for subj, rel, obj in context:
            lines.append(f"   • {subj} {rel} {obj}")
        
        return '\n'.join(lines)
    
    def get_neighbors(self, entity: str, hops: int = 1) -> Dict[int, List[str]]:
        """Lấy danh sách neighbor trong phạm vi hops"""
        if entity not in self.graph.nodes:
            return {}
        
        result = {0: [entity]}
        visited = {entity}
        current = [entity]
        
        for depth in range(1, hops + 1):
            next_level = []
            for node in current:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            result[depth] = next_level
            current = next_level
        
        return result
    
    def explain_path(self, source: str, target: str) -> str:
        """Tìm và giải thích đường đi giữa 2 entity"""
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            if len(path) == 2:
                rel = self.graph[source][target]['relation']
                return f"📌 {source} --{rel}--> {target}"
            else:
                explanations = []
                for i in range(len(path) - 1):
                    rel = self.graph[path[i]][path[i+1]]['relation']
                    explanations.append(f"{path[i]} {rel} {path[i+1]}")
                return f"📌 Path: {' → '.join(explanations)}"
        except nx.NetworkXNoPath:
            return f"❌ No path found between {source} and {target}"
        except nx.NodeNotFound:
            return f"❌ One or both entities not found in graph"
