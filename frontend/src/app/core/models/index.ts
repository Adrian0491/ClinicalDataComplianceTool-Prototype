// ── Auth ─────────────────────────────────────────────────────────────────────
export interface LoginRequest  { email: string; password: string; }
export interface TokenResponse { access_token: string; refresh_token: string; token_type: string; }
export interface MfaLoginResponse { mfa_required: true; temp_token: string; }
export type LoginResponse = TokenResponse | MfaLoginResponse;
export interface MfaVerifyRequest { temp_token: string; totp_code: string; }
export interface RefreshRequest   { refresh_token: string; }
export interface MfaSetupResponse { secret: string; provisioning_uri: string; }

// ── User ──────────────────────────────────────────────────────────────────────
export type Role = 'super_admin' | 'tenant_admin' | 'validator' | 'viewer';

export interface TokenPayload {
  sub: string;
  type: string;
  tenant_id: string;
  role: Role;
  jti: string;
  iat: number;
  exp: number;
}

// ── Study ─────────────────────────────────────────────────────────────────────
export type StudyStatus = 'active' | 'locked' | 'archived';

export interface Study {
  id: string;
  tenant_id: string;
  study_id: string;
  title: string | null;
  phase: string | null;
  therapeutic_area: string | null;
  sponsor: string | null;
  status: StudyStatus;
  created_at: string;
}

export interface StudyCreate {
  study_id: string;
  title?: string;
  phase?: string;
  therapeutic_area?: string;
  sponsor?: string;
}

export interface StudyUpdate {
  title?: string;
  phase?: string;
  therapeutic_area?: string;
  sponsor?: string;
  status?: StudyStatus;
}

// ── Dataset ───────────────────────────────────────────────────────────────────
export type FileFormat = 'csv' | 'json' | 'dataset-json';
export type Domain     = 'DM' | 'VS' | 'AE' | 'CM' | 'EX' | 'LB' | 'DS' | 'EG' | 'QS' | 'RS' | 'PR' | 'MULTI' | 'DATASET_JSON' | string;

export interface Dataset {
  id: string;
  tenant_id: string;
  study_id: string;
  domain: Domain;
  filename: string;
  storage_uri: string;
  file_format: FileFormat;
  row_count: number | null;
  created_at: string;
}

// ── Validation ────────────────────────────────────────────────────────────────
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface ValidationJob {
  id: string;
  tenant_id: string;
  study_id: string;
  status: JobStatus;
  rule_profile: string;
  dataset_ids: string[];
  celery_task_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  created_at: string;
}

export interface ValidationRunRequest {
  study_id: string;
  dataset_ids: string[];
  rule_profile?: string;
}

export interface ValidationSummary {
  job_id: string;
  total: number;
  CRIT: number;
  HIGH: number;
  MED: number;
  LOW: number;
  domains: Record<string, number>;
}

// ── Finding ───────────────────────────────────────────────────────────────────
export type Severity     = 'CRIT' | 'HIGH' | 'MED' | 'LOW';
export type FindingStatus = 'open' | 'resolved' | 'waived';

export interface Finding {
  id: string;
  job_id: string;
  study_id: string;
  finding_type: string;
  rule_id: string;
  severity: Severity;
  domain: string;
  field: string;
  message: string;
  row_index: number;
  usubjid: string | null;
  evidence: string | null;
  status: FindingStatus;
  resolved_by: string | null;
  resolved_at: string | null;
  resolution_note: string | null;
  created_at: string;
}

export interface FindingFilters {
  study_id?: string;
  job_id?: string;
  domain?: string;
  severity?: Severity;
  status?: FindingStatus;
  usubjid?: string;
  offset?: number;
  limit?: number;
}

export interface FindingResolveRequest {
  status: 'resolved' | 'waived';
  resolution_note?: string;
}

// ── Report ────────────────────────────────────────────────────────────────────
export interface ComplianceReport {
  id: string;
  tenant_id: string;
  job_id: string;
  study_id: string;
  report_type: string;
  storage_uri: string | null;
  generated_by: string | null;
  signed_at: string | null;
  signed_by: string | null;
  created_at: string;
  updated_at: string;
}

// ── Audit ─────────────────────────────────────────────────────────────────────
export interface AuditLog {
  id: string;
  tenant_id: string | null;
  user_id: string | null;
  action: string;
  target_type: string | null;
  target_id: string | null;
  ip_address: string | null;
  before_hash: string | null;
  after_hash: string | null;
  occurred_at: string;
}

// ── Shared ────────────────────────────────────────────────────────────────────
export interface ApiError {
  detail: string | { msg: string; type: string }[];
}

export interface MessageResponse { message: string; }
